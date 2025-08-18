from utils.metrics import *
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class Metrics:
    @staticmethod
    def s_dice_score(pred: torch.Tensor,
                    target: torch.Tensor,
                    num_classes: Optional[int] = None,
                    smooth: float = 1e-5
                ) -> torch.Tensor:
        return dice_score(pred, target, num_classes, smooth)

    @staticmethod
    def s_jaccard_index(pred: torch.Tensor,
                        target: torch.Tensor,
                        num_classes: Optional[int] = None,
                        smooth: float = 1e-5
                    ) -> torch.Tensor:
        if pred.dim() == 4:  # [B, D, H, W]
            # Convert to one-hot if needed
            if num_classes is None:
                num_classes = int(max(pred.max(), target.max()) + 1)
            
            pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
            target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Compute intersection and union
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        union = union - intersection
        union = torch.clamp(union, min=smooth)
        jaccard = (intersection + smooth) / union
        return jaccard.mean(dim=1)

    @staticmethod
    def s_hausdorff_distance(seg1: torch.Tensor,
                            seg2: torch.Tensor,
                            spacing: Optional[Tuple[float, float, float]] = None,
                            percentile: float = 95.0
                        ) -> float:
        kernel = torch.ones(1, 1, 3, 3, 3, device=seg1.device)

        # Erode to find boundaries
        seg1_float = seg1.float().unsqueeze(0).unsqueeze(0)
        seg2_float = seg2.float().unsqueeze(0).unsqueeze(0)
        
        seg1_eroded = F.conv3d(seg1_float, kernel, padding=1) < kernel.sum()
        seg2_eroded = F.conv3d(seg2_float, kernel, padding=1) < kernel.sum()
        
        surface1 = seg1_float & ~seg1_eroded
        surface2 = seg2_float & ~seg2_eroded
        
        # Get surface point coordinates
        coords1 = torch.nonzero(surface1.squeeze())
        coords2 = torch.nonzero(surface2.squeeze())
        
        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0
        
        # Apply spacing if provided
        if spacing is not None:
            spacing_tensor = torch.tensor(spacing, device=seg1.device)
            coords1 = coords1.float() * spacing_tensor
            coords2 = coords2.float() * spacing_tensor
        
        # Compute pairwise distances
        # This can be memory intensive for large surfaces
        if len(coords1) * len(coords2) > 1e8:
            # Subsample for large surfaces
            stride1 = max(1, len(coords1) // 10000)
            stride2 = max(1, len(coords2) // 10000)
            coords1 = coords1[::stride1]
            coords2 = coords2[::stride2]
        
        # Compute distances from surface1 to surface2
        dists_1to2 = torch.cdist(coords1.float(), coords2.float())
        min_dists_1to2 = dists_1to2.min(dim=1)[0]
        
        # Compute distances from surface2 to surface1
        dists_2to1 = dists_1to2.t()
        min_dists_2to1 = dists_2to1.min(dim=1)[0]
        
        # Combine distances
        all_dists = torch.cat([min_dists_1to2, min_dists_2to1])
        return torch.quantile(all_dists, percentile).item()
        
    @staticmethod
    def s_local_normalized_cross_correlation(fixed: torch.Tensor,
                                            warped: torch.Tensor,
                                            window_size: int = 9,
                                            eps: float = 1e-8
                                        ) -> torch.Tensor:
        return normalized_cross_correlation(fixed, warped, window_size, eps)

    @staticmethod
    def s_normalized_cross_correlation(
        fixed: torch.Tensor,
        warped: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        B = fixed.shape[0]

        # Flatten spatial dimensions
        fixed_flat = fixed.view(B, -1)
        warped_flat = warped.view(B, -1)

        # Compute global means
        fixed_mean = fixed_flat.mean(dim=1, keepdim=True)
        warped_mean = warped_flat.mean(dim=1, keepdim=True)

        # Compute global variance and covariance
        fixed_var = ((fixed_flat - fixed_mean) ** 2).mean(dim=1)
        warped_var = ((warped_flat - warped_mean) ** 2).mean(dim=1)
        covar = ((fixed_flat - fixed_mean) * (warped_flat - warped_mean)).mean(dim=1)

        # Compute NCC
        ncc = covar / (torch.sqrt(fixed_var * warped_var) + eps)

        return ncc

    @staticmethod
    def s_squared_error(pred: torch.Tensor,
                        target: torch.Tensor
                    ) -> torch.Tensor:
        if pred.dim() == 4:
            # Convert to one-hot if needed
            num_classes = int(max(pred.max(), target.max()) + 1)
            pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
            target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        # Compute squared error
        squared_error = (pred - target) ** 2
        return squared_error.mean(dim=(2, 3, 4))

    @staticmethod
    def s_signal_to_noise_ratio(pred, target):
        pass

    @staticmethod
    def s_structural_similarity_index(img1: torch.Tensor,
                                        img2: torch.Tensor,
                                        window_size: int = 11,
                                        K1: float = 0.01,
                                        K2: float = 0.03,
                                        data_range: float = 1.0
                                    ) -> torch.Tensor:
        return structural_similarity_index(img1, img2, window_size, K1, K2, data_range)

    @staticmethod
    def s_mutual_information(fixed: torch.Tensor,
                       warped: torch.Tensor,
                       num_bins: int = 64,
                       eps: float = 1e-8) -> torch.Tensor:
        B = fixed.shape[0]
        mi_vals = []

        for b in range(B):
            # Flatten images
            f = fixed[b].view(-1)
            w = warped[b].view(-1)
            
            # Compute min/max for histogram binning
            f_min, f_max = f.min(), f.max()
            w_min, w_max = w.min(), w.max()
            
            # Compute histograms
            f_hist = torch.histc(f, bins=num_bins, min=f_min, max=f_max)
            w_hist = torch.histc(w, bins=num_bins, min=w_min, max=w_max)
            
            # Joint histogram
            f_bin = torch.clamp(((f - f_min) / (f_max - f_min) * (num_bins - 1)).long(), 0, num_bins - 1)
            w_bin = torch.clamp(((w - w_min) / (w_max - w_min) * (num_bins - 1)).long(), 0, num_bins - 1)
            
            joint_hist = torch.zeros((num_bins, num_bins), device=f.device)
            joint_hist.index_put_((f_bin, w_bin), torch.ones_like(f_bin, dtype=torch.float), accumulate=True)
            
            # Convert to probabilities
            p_f = f_hist / f_hist.sum()
            p_w = w_hist / w_hist.sum()
            p_fw = joint_hist / joint_hist.sum()
            
            # Compute MI
            p_f_expand = p_f.view(-1, 1)
            p_w_expand = p_w.view(1, -1)
            
            mi_matrix = p_fw * torch.log((p_fw + eps) / (p_f_expand * p_w_expand + eps))
            mi = mi_matrix.sum()
            mi_vals.append(mi)
        
        return torch.stack(mi_vals)
    
    @staticmethod
    def s_normalized_mutual_information(fixed: torch.Tensor,
                                    warped: torch.Tensor,
                                    num_bins: int = 64,
                                    eps: float = 1e-8) -> torch.Tensor:
        B = fixed.shape[0]
        nmi_vals = []

        for b in range(B):
            f = fixed[b].view(-1)
            w = warped[b].view(-1)
            
            f_min, f_max = f.min(), f.max()
            w_min, w_max = w.min(), w.max()
            
            # Compute histograms
            f_hist = torch.histc(f, bins=num_bins, min=f_min, max=f_max)
            w_hist = torch.histc(w, bins=num_bins, min=w_min, max=w_max)
            
            f_bin = torch.clamp(((f - f_min) / (f_max - f_min) * (num_bins - 1)).long(), 0, num_bins - 1)
            w_bin = torch.clamp(((w - w_min) / (w_max - w_min) * (num_bins - 1)).long(), 0, num_bins - 1)
            
            joint_hist = torch.zeros((num_bins, num_bins), device=f.device)
            joint_hist.index_put_((f_bin, w_bin), torch.ones_like(f_bin, dtype=torch.float), accumulate=True)
            
            # Convert to probabilities
            p_f = f_hist / f_hist.sum()
            p_w = w_hist / w_hist.sum()
            p_fw = joint_hist / joint_hist.sum()
            
            # Mutual information
            p_f_expand = p_f.view(-1, 1)
            p_w_expand = p_w.view(1, -1)
            mi_matrix = p_fw * torch.log((p_fw + eps) / (p_f_expand * p_w_expand + eps))
            mi = mi_matrix.sum()
            
            # Entropies
            h_f = -torch.sum(p_f * torch.log(p_f + eps))
            h_w = -torch.sum(p_w * torch.log(p_w + eps))
            
            # Normalized mutual information
            nmi = mi / torch.sqrt(h_f * h_w + eps)
            nmi_vals.append(nmi)
        
        return torch.stack(nmi_vals)

    @staticmethod
    def s_intersection_over_union(pred: torch.Tensor,
                        target: torch.Tensor,
                        num_classes: Optional[int] = None,
                        smooth: float = 1e-5
                    ) -> torch.Tensor:
        # same as jaccard_index
        return Metrics.s_jaccard_index(pred, target, num_classes, smooth)

    @staticmethod
    def s_tre(landmarks_fixed: torch.Tensor,
                landmarks_moving: torch.Tensor,
                displacement: torch.Tensor,
                spacing: Optional[Tuple[float, float, float]] = None
            ) -> torch.Tensor:
        return target_registration_error(landmarks_fixed, landmarks_moving, displacement, spacing)

    @staticmethod
    def s_absolute_error(pred: torch.Tensor,
                        target: torch.Tensor
                    ) -> torch.Tensor:
        if pred.dim() == 4:
            # Convert to one-hot if needed
            num_classes = int(max(pred.max(), target.max()) + 1)
            pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
            target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        # Compute absolute error
        absolute_error = torch.abs(pred - target)
        return absolute_error.mean(dim=(2, 3, 4))

    @staticmethod
    def s_mean_surface_distance(seg1: torch.Tensor,
                        seg2: torch.Tensor,
                        spacing: Optional[Tuple[float, float, float]] = None
                    ) -> float:
        surface_distances = surface_distance(seg1, seg2, spacing)
        return surface_distances['mean_surface_distance']

    @staticmethod
    def _s_jacobian_determinant(displacement_field: torch.Tensor) -> torch.Tensor:
        return jacobian_determinant(displacement_field)

    @staticmethod
    def s_std_jacobian(displacement_field: torch.Tensor) -> float:
        return Metrics._s_jacobian_determinant(displacement_field).std().item()

    @staticmethod
    def s_mean_jacobian(displacement_field: torch.Tensor) -> float:
        return Metrics._s_jacobian_determinant(displacement_field).mean().item()
    
    @staticmethod
    def s_negative_jacobian(displacement_field: torch.Tensor) -> float:
        return (Metrics._s_jacobian_determinant(displacement_field)<= 0).float().mean().item()
