from common_import import *
from src.utils import complex_gaussian_matrix

def main(iterations=10,
         snr=20,
         rho=1,
         cost = 1.0,
         device= "cpu"):
    
    user_languages = ["vit_small_patch32_224", "vit_small_patch16_224"]
    user = {idx : Agent(dataset="cifar10", language_name=user_languages[idx], device=device) for idx in range(len(user_languages))}
    bs = Base_Station(dataset="cifar10", model_name="vit_base_patch16_224")
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(bs.antennas_transmitter, bs.antennas_transmitter)) 
    #pre-whit. signal 
    bs.X, bs.L, bs.mean_X = prewhiten(bs.X, device="cpu")
    _, n = bs.X.shape
    with torch.no_grad():
            
        for i in tqdm(range(iterations), "Computing ADMM scaled optimisation"):
            for k in range(len(user_languages)):
                #----------G STEP----------
                HFX = bs.AP_message_broadcast(channel_matrix = channel_matrix)
                sigma = sigma_given_snr(snr, torch.ones(1)/math.sqrt(bs.antennas_transmitter))
                #G_k = Y(HFX)((HFX)(HFX)+nKΣ)^-1
                user[k].G_k = (user[0].Y @ HFX.H @ torch.linalg.inv(HFX @ HFX.H + n * sigma * torch.view_as_complex(torch.stack((torch.eye(HFX.shape[0]), torch.eye(HFX.shape[0])), dim=-1)).to(device))).to(device)
                assert user[k].G_k.shape == ((user[k].train_user.latent_space.shape[-1] + 1) // 2, user[k].antennas_receiver), f"Expected G_k shape has dimension issues"
            
                #----------F_k STEP----------
            rho_n = rho * n
            for k in range(len(user_languages)):
                bs.bs_buffer = user[k].user_message_broadcast(channel_matrix = channel_matrix) #a list that contain for each idx a list of A and (GH)^H(Y)
                A = bs.bs_buffer[0]
                B = rho_n * torch.linalg.inv(bs.X @ bs.X.H)
                C = (rho_n * (bs.Z - bs.U)) + (bs.bs_buffer[1] @ bs.X.H) @ (B/(rho_n)) #quest'ultimo termine va controllato 
                bs.F_k[k] = torch.tensor(solve_sylvester(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()), device=device)
                
            bs._F_aggregation()
            
              #----------Dual STEP----------
            C = bs.F + bs.U
            tr = torch.trace(C @ C.H).real
            
            if tr <= cost:
                bs.Z = C
            else:
                lmb = torch.sqrt(tr / cost).item() -1
                bs.Z = C / (1 + lmb)
                
            bs.U = bs.U + bs.F - bs.Z
            admm_residuals = torch.linalg.matrix_norm(bs.F - bs.Z)
            print(f"Residuals at iter {i} is: {admm_residuals}")

if __name__=="__main__":
    main()       
        