from .models import SpikeRegUNet
import torch
import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt
from .registration import SpikeRegInference
from .registration import IterativeRegistration
from .models import PretrainedUNet
from .training import SpikeRegTrainer

from .utils.warping import SpatialTransformer
import yaml
import os

import plotly.graph_objects as go
import torch

def load_model_weights(model, checkpoint_path, device='cpu'):
    """Load weights from checkpoint into model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', None)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    print(config)
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model = SpikeRegUNet(config['model']).to(device)
    
    model.load_state_dict(state_dict)
    return model, config

def get_inference():
    """Run inference using SpikeReg model directly."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load images
    print("Loading images...")
    # img = nib.load('data/L2R_2021_Task3_val/img0454.nii.gz')
    # img_data = img.get_fdata()
    # img_data = img_data.astype(np.float32)
    # img_tensor = torch.from_numpy(img_data).unsqueeze(0)

    # moving = nib.load('data/L2R_2021_Task3_val/img0455.nii.gz')
    # moving_data = moving.get_fdata()
    # moving_data = moving_data.astype(np.float32)
    # moving_tensor = torch.from_numpy(moving_data).unsqueeze(0)
    # import nibabel as nib
    # import numpy as np
    # import torch

    print("Loading images...")

    # Load fixed image
    img = nib.load('data/L2R_2021_Task3_val/img0454.nii.gz')
    img_data = img.get_fdata().astype(np.float32)
    img_tensor = torch.from_numpy(img_data).unsqueeze(0)  # [1, D, H, W]

    # Load moving image
    moving = nib.load('data/L2R_2021_Task3_val/img0456.nii.gz')
    moving_data = moving.get_fdata().astype(np.float32)
    moving_tensor = torch.from_numpy(moving_data).unsqueeze(0)  # [1, D, H, W]

    # Stack to create a batch of size 8 (same images repeated)
    # Resulting shape: [8, 1, D, H, W]
    img_tensor = img_tensor.repeat(1,1, 1, 1, 1)
    moving_tensor = moving_tensor.repeat(1,1, 1, 1, 1)

    print("img_batch shape:", img_tensor.shape)
    print("moving_tensor shape:", moving_tensor.shape)


    print(f"Image shapes: fixed {img_tensor.shape}, moving {moving_tensor.shape}")

    # Load config and create model
    config_path = 'checkpoints/oasis/config.yaml'
    model_path = 'checkpoints/oasis/final_model.pth'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model and load weights
    # model = PretrainedUNet(config['model']).to(device)
    # model = SpikeRegUNet(config['model']).to(device)
    model, config_model = load_model_weights(None, model_path, device)
    config = config_model if config_model is not None else config
    model.eval()
    # model_state_dict=torch.load('checkpoints/oasis/final_model.pth', weights_only=False,map_location=torch.device('cpu'))
    # print(model_state_dict['model_state_dict'].keys())
    # return


    # Prepare inputs
    fixed = img_tensor.to(device)  # Add batch dimension
    moving = moving_tensor.to(device)

    # Initialize spatial transformer
    spatial_transformer = SpatialTransformer().to(device)

    # Create registration wrapper
    # registration = SpikeRegInference(
    #     model_path=model_path,
    #     # config=config,
    #     device=device,
    #     patch_size=config['model'].get('patch_size', 32),
    #     patch_stride=config['model'].get('patch_stride', 16),
    #     batch_size=1
    # )

    registration = IterativeRegistration(
        model,
        num_iterations=10,
        early_stop_threshold=0.001,
    )



    # Run inference
    # # import torch
    # ckpt = torch.load('checkpoints/oasis/final_model.pth', map_location='cpu')
    # state = ckpt.get('model_state_dict', ckpt)
    # for k,v in state.items():
    #     if hasattr(v, 'shape') and tuple(v.shape) == (128,128,2,2,2):
    #         print('module expecting 128 in-channels:', k)


    # from SpikeReg.registration import SpikeRegInference  # or .registration import SpikeRegInference

    # infer = SpikeRegInference(
    #     model_path='checkpoints/oasis/final_model.pth',
    #     config=config,
    #     device='cpu'  # or 'cuda'
    # )

    # # inputs can be numpy volumes or tensors; the class will preprocess
    # print("shape: ", img_data.shape)
    # displacement_np = infer.register(img_data, moving_data)  # returns displacement numpy array
    # # If you want the warped image:
    # warped = infer.apply_deformation(moving_data, displacement_np)
    # return

    # sd = torch.load('checkpoints/oasis/final_model.pth', map_location='cpu')
    # state = sd['state_dict'] if 'state_dict' in sd else sd
    # print("state params")

    # trainer = SpikeRegTrainer(config, None, None, device=device)
    # trainer.load_checkpoint('final_model.pth')

    # trainer.convert_to_spiking()
    
    # for k,v in state['model_state_dict'].items():
    #     try:    
    #         print(k, v.shape)
    #     except:
    #         # pass
    #         print(k, v)
    # # Then compare to your model's named_parameters()
    # print('\n\nmodel params:')
    # for k,p in trainer.spiking_model.named_parameters():
    #     print(k, p.shape)

    # model = trainer.spiking_model
    registration = IterativeRegistration(
        model,
        num_iterations=10,
        early_stop_threshold=0.001,
    )
    # return

    with torch.no_grad():
        print("Running registration...")
        print(config)
        output = registration(fixed, moving) if isinstance(model, SpikeRegUNet) else registration(fixed, moving)
        print("Registration output obtained.")
        # output = model(fixed, moving)
        print(output)
        displacement = output['displacement'].to(device) if True else output
        print("Displacement field obtained.")
        
        # Apply transformation
        warped = spatial_transformer(moving, displacement)
    print(warped.shape)

    warped_np = warped.cpu().numpy()
    displacement_np = displacement.cpu().numpy()
    # print(displacement_np)
    
    warped_nii = nib.Nifti1Image(warped_np[0, 0], img.affine, img.header)
    displacement_nii = nib.Nifti1Image(displacement_np[0], img.affine, img.header)

    # Call function to compute and save metrics and visualization
    show_results(device, img, img_data, moving, moving_data, fixed, displacement, warped, warped_nii, displacement_nii)

def show_results(device, img, img_data, moving, moving_data, fixed, displacement, warped, warped_nii, displacement_nii):
    os.makedirs('outputs', exist_ok=True)
    nib.save(warped_nii, 'outputs/warped.nii.gz')
    nib.save(displacement_nii, 'outputs/displacement.nii.gz')

    # Calculate comprehensive metrics
    from .utils.metrics import compute_registration_metrics

    # Add channel dimension if needed
    fixed_tensor = fixed.squeeze().unsqueeze(0).unsqueeze(0) if fixed.squeeze().dim() == 3 else fixed
    warped_tensor = warped.squeeze().unsqueeze(0).unsqueeze(0) if warped.squeeze().dim() == 3 else warped
    moving_tensor = moving_data if isinstance(moving_data, torch.Tensor) else torch.from_numpy(moving_data)
    moving_tensor = moving_tensor.squeeze().unsqueeze(0).unsqueeze(0).to(device) if moving_tensor.squeeze().dim() == 3 else moving_tensor.to(device)

    # Get image spacing from header if available
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3]) if hasattr(img, 'header') else None

    # Compute all available metrics
    with torch.no_grad():
        metrics = compute_registration_metrics(
            fixed=fixed_tensor,
            moving=moving_tensor,
            warped=warped_tensor,
            displacement=displacement,
            spacing=spacing
        )
    
    # Print metrics
    print("\nRegistration Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n{metric_name}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
    print()

    # Compute Dice score if segmentations are available
    try:
        # Construct segmentation filenames
        fixed_seg_path = img.get_filename().replace('img', 'seg')
        moving_seg_path = moving.get_filename().replace('img', 'seg')
        
        if os.path.exists(fixed_seg_path) and os.path.exists(moving_seg_path):
            # Load segmentations
            fixed_seg = nib.load(fixed_seg_path)
            moving_seg = nib.load(moving_seg_path)
            fixed_seg_data = fixed_seg.get_fdata().astype(np.int32)
            moving_seg_data = moving_seg.get_fdata().astype(np.int32)
            
            # Convert to tensors
            fixed_seg_tensor = torch.from_numpy(fixed_seg_data).unsqueeze(0).unsqueeze(0)
            moving_seg_tensor = torch.from_numpy(moving_seg_data).unsqueeze(0).unsqueeze(0)
            
            # Create segmentation transformer with nearest neighbor interpolation
            seg_transformer = SpatialTransformer(mode='nearest', padding_mode='border').to(device)
            
            # Warp moving segmentation
            warped_seg = seg_transformer(
                moving_seg_tensor.float().to(device), 
                displacement
            ).long()
            
            # Get unique labels (excluding background 0)
            labels = torch.unique(fixed_seg_tensor)
            labels = labels[labels > 0]
            
            dice_scores = {}
            mean_dice = 0.0
            num_valid_labels = 0
            
            # Compute Dice for each label
            for label in labels:
                fixed_mask = (fixed_seg_tensor == label).float()
                warped_mask = (warped_seg == label).float()
                
                intersection = (fixed_mask * warped_mask).sum()
                total = fixed_mask.sum() + warped_mask.sum()
                
                if total > 0:  # Avoid division by zero
                    dice = 2.0 * intersection / total
                    dice_scores[f'dice_label_{int(label)}'] = float(dice)
                    mean_dice += float(dice)
                    num_valid_labels += 1
            
            # Compute mean Dice across labels
            if num_valid_labels > 0:
                mean_dice = mean_dice / num_valid_labels
                dice_scores['dice_mean'] = mean_dice
                
                # Print Dice scores
                print("\nDice Scores:")
                for label, score in dice_scores.items():
                    print(f"{label}: {score:.4f}")
                
                # Save warped segmentation
                warped_seg_np = warped_seg.cpu().numpy()
                warped_seg_nii = nib.Nifti1Image(warped_seg_np[0, 0], img.affine, img.header)
                nib.save(warped_seg_nii, 'outputs/warped_seg.nii.gz')
    except Exception as e:
        print(f"\nSkipping Dice computation: {str(e)}")
        dice_scores = {}

    # Save metrics to file
    # Convert all values to native Python types for YAML serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable_metrics[key] = {k: float(v) for k, v in value.items()}
        else:
            serializable_metrics[key] = float(value)
    
    with open('outputs/metrics.yaml', 'w') as f:
        yaml.dump(serializable_metrics, f, default_flow_style=False)

    # Visualize middle slices
    try:
        import matplotlib.pyplot as plt
        
        warped_vis = warped.squeeze()  # Remove batch and channel dims
        middle_slice = warped_vis.shape[0] // 2
        slice1 = warped_vis[middle_slice].cpu()
        slice2 = img_data[middle_slice]
        slice3 = moving_data[middle_slice]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(slice1.T, cmap='gray', origin='lower')
        axes[0].set_title(f'Warped middle slice ({middle_slice})')
        axes[0].axis('off')

        axes[1].imshow(slice2.T, cmap='gray', origin='lower')
        axes[1].set_title(f'Fixed middle slice ({middle_slice})')
        axes[1].axis('off')

        axes[2].imshow(slice3.T, cmap='gray', origin='lower')
        axes[2].set_title(f'Moving middle slice ({middle_slice})')
        axes[2].axis('off')

        plt.tight_layout()
        
        # Save the visualization
        plt.savefig('outputs/registration_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nVisualization saved as 'outputs/registration_visualization.png'")
###################
        # Convert and normalize
        vol = warped.squeeze().cpu().numpy()
        # vol = (vol - vol.min()) / (vol.max() - vol.min())

        # --- Reduce size ---
        # Downsample by factor of 4 (adjust as needed: 2 = better quality, 8 = smaller)
        factor = 2
        vol_small = vol[::factor, ::factor, ::factor]

        # Create coordinate grid (much smaller now)
        x, y, z = np.mgrid[0:vol_small.shape[0],
                        0:vol_small.shape[1],
                        0:vol_small.shape[2]]

        # Build figure
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=vol_small.flatten(),
            isomin=0.3,
            isomax=1.0,
            opacity=0.1,          # transparency
            surface_count=6,      # number of contour surfaces
            colorscale='gray'
        ))

        fig.update_layout(
            title='3D Volume (Warped, downsampled)',
            scene_aspectmode='cube',
            width=600,  # reduce figure size
            height=600
        )

        fig.write_html("outputs/warped_volume_3d.html")

    except ImportError:
        print("\nSkipping visualization: matplotlib not available")
    except Exception as e:
        print(f"\nVisualization failed: {str(e)}")