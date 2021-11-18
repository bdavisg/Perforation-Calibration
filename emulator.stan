// Fits a single observed bolometric light curve with an ensemble of
// gaussian processes that predict predict principal component
// coefficients.

// Author: Danny Goldstein

functions {

  real nickel_mass_dist_log(real y){
    return log(2 * ((-1.0625 * y) + 1.2625));
  }

}
  
data {
  
  int<lower=1> D; // Number of points in the training light curves  
  int<lower=1> N; // Number of GPs, Number of PCs
  int<lower=1> T; // Number of training examples
  int<lower=1> V; // Dimensionality of simulation input parameter
		  // space

  vector[V] x_train_scaled[T]; // Training inputs scaled according to
			       // x_train_scaled = (x_train - mu_x) /
			       // (sigma_x)

  vector[V] mu_x;
  vector[V] sigma_x;                                                              

  vector[D] mu_L; // Mean of training light curve vectors
  
  vector[D] sigma_L; // Standard deviation of training light curve
		     // vectors
  
  vector[D] L_train_scaled[T]; // Training responses scaled by
			       // L_train_scaled = (L_train - mu_L) /
			       // (sigma_L)

  
  matrix[N, D] principal_components; // Principal components of scaled
				     // training light curves

  vector[N] mu_pc_coeffs_train; // Mean of principal component
				// coefficient vectors for training
				// LCs
  
  vector[N] sigma_pc_coeffs_train; // Standard deviation of principal
				   // component coefficient vectors
				   // for training LCs


  vector[T] pc_coeffs_train_scaled[N]; // Principal component
				       // coefficients for training
				       // LCs scaled according to
				       // pc_coeffs_train_scaled =
				       // (pc_coeffs_train -
				       // mu_pc_coeffs_train) /
				       // sigma_pc_coeffs_train
 
  vector[D] L_star; // Observed target LC. (LC to fit).
  vector[D] L_star_uncertainty; // Uncertainty on L_star.
  
  real nug; // Nugget. 
}

parameters {

  // Sampling Parameters
           
  real<lower=0.2,  upper=1.0>    nickel_mass; 
  real<lower=0.0,  upper=1.0>    ejecta_par;  
  real<lower=0.05, upper=0.18>   kappa;       
  real<lower=0.00, upper=0.15>   unburned_mass;
  real<lower=0.0,  upper=1.0>    mixing_par;  

  vector[N] pc_coeffs_realized_scaled;  

  // Hyperparameters
  
  vector<lower=0>[N] w;
  real<lower=0> rho_sq;

}

transformed parameters {
           
  vector[D] L_realized;
  vector[V] x_star;
  vector[V] x_star_scaled;

  // Sampling --> Inference              
              
  real nickel_plus_co;
  real ejecta_mass;
  real ime_mass;
  real nickel_radius;

  nickel_plus_co <- nickel_mass + unburned_mass;
  ejecta_mass <- nickel_plus_co + ejecta_par * (1.38 - nickel_plus_co);
  if (ejecta_mass < 0.8)
    ejecta_mass <- 0.8 + ejecta_par * (1.38 - 0.8);
  ime_mass <- ejecta_mass - nickel_plus_co;
  nickel_radius <- mixing_par * (ejecta_mass - nickel_mass) + nickel_mass;

  // Inference parameters. 
                   
  x_star[1] <- nickel_mass;
  x_star[2] <- ime_mass;
  x_star[3] <- unburned_mass;
  x_star[4] <- nickel_radius;
  x_star[5] <- kappa;

  L_realized <- (to_vector(to_row_vector((pc_coeffs_realized_scaled .* sigma_pc_coeffs_train)
                                          +   mu_pc_coeffs_train) * principal_components) 
                                          .* sigma_L) + mu_L;

  x_star_scaled <- (x_star - mu_x) ./ sigma_x;
}

model {

  matrix[T+1,T+1] Sigma; // Unconditioned covariance matrix  
  matrix[N, T+1] y; // Sandwiched scaled pc coefficients.
  vector[V] x[T+1];

  for (i in 1:T)
    x[i] <- x_train_scaled[i];
  x[T+1] <- x_star_scaled;
  
  for (i in 1:N){
    for (j in 1:T){
      y[i, j] <- pc_coeffs_train_scaled[i, j];
    }
    y[i, T + 1] <- pc_coeffs_realized_scaled[i];
  }


  for (i in 1:T){
    for (j in i+1:T+1){
      Sigma[i, j] <- exp(-rho_sq * dot_self(x[i] - x[j]));
      Sigma[j, i] <- Sigma[i, j];
    }
  }

  for (i in 1:T+1)
    Sigma[i, i] <- 1 + nug;
  
  y                  ~ multi_gp(Sigma, w);
  L_realized         ~ normal(L_star, L_star_uncertainty);
  ime_mass           ~ normal(.58, .45) T[0, 1.38];
  nickel_radius      ~ normal(0.8, .4) T[0, 1.38];
  nickel_mass        ~ nickel_mass_dist();

  print(L_realized);

}
