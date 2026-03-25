import torch

def simple_diffusion_step(model, x_t, t_i, t_im1, alphas_cumprod, eta=0.0):
    """
    Réalise une seule étape de reverse diffusion DDIM standard inconditionnelle.
    Utilisé pour terminer la diffusion sans appliquer les contraintes analytiques.
    """
    alpha_bar_t = alphas_cumprod[t_i]
    alpha_bar_prev = alphas_cumprod[t_im1] if t_im1 >= 0 else torch.tensor(1.0, device=x_t.device)
    
    # Convert timesteps to tensor if they are integers/floats
    if not isinstance(t_i, torch.Tensor):
        t_i_tensor = torch.tensor([t_i], device=x_t.device)
    else:
        t_i_tensor = t_i.view(-1).to(x_t.device)

    # 1. Prédiction de l'erreur
    with torch.no_grad():
        model_out = model(x_t, t_i_tensor)
        # Si le modèle prédit la variance, il retourne 6 canaux, on garde les 3 premiers
        eps_pred = model_out[:, :3, :, :] if model_out.shape[1] == 6 else model_out
    
    # 2. Estimation de x_0 (Tweedie)
    sqrt_at = torch.sqrt(alpha_bar_t)
    sqrt_1mat = torch.sqrt(1 - alpha_bar_t)
    x0_hat = (x_t - sqrt_1mat * eps_pred) / sqrt_at

    # 3. Saut DDIM vers t-1
    # On calcule la variance pour ce pas DDIM (si eta=0, c'est purement déterministe)
    # Formule : sigma_t = eta * sqrt((1 - alpha_prev)/(1 - alpha_t)) * sqrt(1 - alpha_t/alpha_prev)
    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
    
    # Direction pointant vers x_t
    dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_pred
    
    # Bruit aléatoire si on n'est pas à la dernière étape
    noise = torch.randn_like(x_t) if t_im1 > 0 and eta > 0 else 0.0
    
    # Équation DDIM réassemblée
    x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma_t * noise

    return x_prev, eps_pred

def single_diffpir_step(x, y, mask, t_i, t_im1, model_fn, rhos, sigmas, alphas_cumprod, guidance_scale, eta=0.0, zeta=0.0, pnp_method='hqs', gamma=1.0):
    # 1. Prédire le bruit (epsilon) via le modèle
    # Résolution de l'hypothèse de la prédiction directe : on suppose que model_fn prédit epsilon
    # (Si votre modèle nécessite "noise_level" ou autre, vous pouvez l'encapsuler dans model_fn)
    if not isinstance(t_i, torch.Tensor):
        t_i_tensor = torch.tensor([t_i], device=x.device)
    else:
        t_i_tensor = t_i.view(-1).to(x.device)
        
    model_out = model_fn(x, t_i_tensor) 
    eps_pred = model_out[:, :3, :, :] if model_out.shape[1] == 6 else model_out

    # Variables de cumul d'alphas
    alpha_bar_t = alphas_cumprod[t_i]
    # Si t_im1 < 0, on se dirige vers l'image finale, alpha_bar_prev = 1.0
    alpha_bar_prev = alphas_cumprod[t_im1] if t_im1 >= 0 else torch.tensor(1.0, device=x.device)

    sqrt_at = torch.sqrt(alpha_bar_t)
    sqrt_1mat = torch.sqrt(1 - alpha_bar_t)

    # Calcul de x0 (Tweedie's formula)
    x0_hat = (x - sqrt_1mat * eps_pred) / sqrt_at
    # x0_hat = x0_hat.clamp(-1, 1) # Recommandé pour la stabilité numérique

    # 2. Correction selon la méthode PnP choisie
    if pnp_method.lower() == 'hqs':
        # --- Half-Quadratic Splitting (HQS) ---
        # Solution analytique exacte pondérée par rho
        x0_p = (mask * y + rhos[t_i] * x0_hat) / (mask + rhos[t_i])
    
    elif pnp_method.lower() == 'pgd':
        # --- Proximal Gradient Descent (PGD) adaptatif façon DPS ---
        # Calcul de base de l'erreur (gradient)
        error = mask * x0_hat - mask * y
        
        # Calcul de la norme L2 de l'erreur pour chaque image du batch
        # Cela permet d'avoir un pas (gamma_t) dynamique inversement proportionnel à l'erreur
        norm_error = torch.linalg.norm(error.reshape(error.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        
        # Pas de gradient dynamique calculé pour ce timestep
        gamma_t = gamma / (norm_error + 1e-6)
        x0_p = x0_hat - gamma_t * error

    else:
        raise ValueError(f"Méthode PnP inconnue: {pnp_method}. Choisissez 'hqs' ou 'pgd'.")

    # On peut appliquer le guidance scale pour doser la force de cette correction
    x0 = x0_hat + guidance_scale * (x0_p - x0_hat)

    # 3. Recalculer le pseudo-epsilon (après ajustement de x0 pour coller à l'observation)
    eps_adjusted = (x - sqrt_at * x0) / sqrt_1mat

    # 4. Saut vers t-1 avec ajout de bruit stochastique (DDIM + Stochasticité DiffPIR)
    # Paramètres de variance
    sqrt_1mat_prev = torch.sqrt(1 - alpha_bar_prev)
    
    # eta_sigma (équivalent à la variance DDIM standard avec terme betas)
    # L'équation du code original est : eta * sqrt_1m[t_im1] / sqrt_1m[t_i] * sqrt(beta[t_i])
    # Note: si beta n'est pas passé, (1 - alpha_bar_t / alpha_bar_prev) est le beta théorique "sauté"
    beta_t_theoretical = 1 - (alpha_bar_t / alpha_bar_prev)
    eta_sigma = eta * (sqrt_1mat_prev / sqrt_1mat) * torch.sqrt(beta_t_theoretical)
    
    # re-calcul de la partie DDIM avec intégration de zeta (\zeta)
    dir_xt = torch.sqrt(torch.clamp(sqrt_1mat_prev**2 - eta_sigma**2, min=0.0)) * eps_adjusted

    noise1 = torch.randn_like(x) if t_im1 > 0 else 0.0
    noise2 = torch.randn_like(x) if t_im1 > 0 else 0.0

    # L'équation exacte du code source :
    x_next = (
        torch.sqrt(alpha_bar_prev) * x0 + 
        torch.sqrt(torch.tensor(1.0 - zeta, device=x.device)) * (dir_xt + eta_sigma * noise1) + 
        torch.sqrt(torch.tensor(zeta, device=x.device)) * sqrt_1mat_prev * noise2
    )
    
    return x_next, x0