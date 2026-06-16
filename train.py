import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, structural_attention_map, weighted_charbonnier_loss
# from pytorch_msssim import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel_Xray
from utils.general_utils import safe_state, gen_log
from tqdm import tqdm
from utils.image_utils import psnr, time2file_name
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
import time
import yaml
import numpy as np
from PIL import Image

from pdb import set_trace as stx


def save_vis_image(tensor, save_path):
    t = tensor.detach().float().cpu()
    if t.dim() == 3:
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.mean(dim=0)
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    t_min, t_max = t.min(), t.max()
    if (t_max - t_min) > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)
    arr = (t.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(save_path)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    exp_logger = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))

    gaussians = GaussianModel_Xray(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    mask_vis_dir = os.path.join(scene.model_path, "mask_vis")
    os.makedirs(mask_vis_dir, exist_ok=True)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.normalized_image.cuda()

        

        Ll1 = l1_loss(image, gt_image)

        if getattr(opt, "use_attention", True):
            attn_struct = structural_attention_map(
                gt_image,
                hf_lambda=opt.attn_hf_lambda,
                blur_kernel=opt.attn_blur_kernel,
                blur_sigma=opt.attn_blur_sigma,
            ).detach()

            if iteration <= opt.attn_warmup_start:
                attn_warmup_ratio = 0.0
            else:
                attn_warmup_ratio = min(1.0, (iteration - opt.attn_warmup_start) / max(1, opt.attn_warmup_iters))

            if getattr(opt, "attn_adaptive_gate", True):
                gate_thresh = torch.quantile(
                    attn_struct.reshape(-1),
                    float(getattr(opt, "attn_gate_quantile", 0.75))
                ).item()
                gate_thresh = max(0.0, min(0.98, gate_thresh))
            else:
                gate_thresh = float(opt.attn_gate_thresh)

            attn_gate = torch.clamp(
                (attn_struct - gate_thresh) / (1.0 - gate_thresh + 1e-6),
                min=0.0,
                max=1.0,
            )
            attn_alpha_eff = opt.attn_alpha * attn_warmup_ratio
            pixel_weight = 1.0 + attn_alpha_eff * attn_gate * attn_struct
        else:
            pixel_weight = torch.tensor(1.0, device="cuda")
            gate_thresh = 0.0 # dummy value for logging

        Lcharb = weighted_charbonnier_loss(image, gt_image, weight=pixel_weight, eps=opt.charb_eps)
        loss = (1.0 - opt.lambda_dssim) * Lcharb + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()


        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 100 == 0:
                exp_logger.info(
                    f"Iter:{iteration}, gate_thresh={gate_thresh:.4f}, Lcharb={Lcharb.item():.6f}"
                )

            if getattr(opt, "mask_vis_interval", 0) > 0 and iteration % opt.mask_vis_interval == 0:
                if getattr(opt, "use_attention", True):
                    save_vis_image(attn_struct, os.path.join(mask_vis_dir, f"iter_{iteration:06d}_attn_struct.png"))
                    save_vis_image(attn_gate, os.path.join(mask_vis_dir, f"iter_{iteration:06d}_attn_gate.png"))
                    save_vis_image(pixel_weight, os.path.join(mask_vis_dir, f"iter_{iteration:06d}_pixel_weight.png"))

            training_report(exp_logger, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)
                
            if iteration in saving_iterations:
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def prepare_output_and_logger(args):    
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.scene, date_time)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    exp_logger = gen_log(args.model_path)

    return exp_logger



def training_report(exp_logger, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, args):
    if exp_logger and (iteration == 0 or (iteration+1) % 100 == 0):
        exp_logger.info(f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    if iteration in testing_iterations:
        torch.cuda.empty_cache()


        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},{'name': 'train', 'cameras' : scene.getTrainCameras()})

        with torch.no_grad():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0 and config['name'] == 'test':
                    psnr_test = 0.0
                    ssim_test = 0.0
                    start = time.time()
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        image = image.mean(dim=0, keepdim=True)
                        gt_image_norm = viewpoint.normalized_image.to("cuda")

                        ssim_test += float(ssim(image, gt_image_norm).mean().item())
                        psnr_test += float(psnr(image, gt_image_norm).mean().item())

                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])

                    end = time.time()
                    exp_logger.info(f"Testing Speed: {len(config['cameras'])/(end-start)} fps")
                    exp_logger.info(f"Testing Time: {end-start} s")
                    exp_logger.info("\n[ITER {}] Evaluating {}: SSIM = {}, PSNR = {}".format(iteration, config['name'], ssim_test, psnr_test))

        if exp_logger:
            exp_logger.info(f'Iter:{iteration}, total_points:{scene.gaussians.get_xyz.shape[0]}')
        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters") 
    lp = ModelParams(parser)                      # 
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/chest.yaml', help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000,])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("Optimizing " + args.model_path)

    safe_state(args.quiet, seed=args.seed, deterministic=True)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")