#' @templateVar MODEL_FUNCTION riggedDeck
#' @templateVar CONTRIBUTOR Mario Martinez-Saito
#' @templateVar TASK_NAME Rigged deck game
#' @templateVar MODEL_NAME Kalman Filter
#' @templateVar MODEL_CITE (Daw et al., 2006, Nature)
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "choice", "outcome"
#' @templateVar PARAMETERS "lambda" (decay factor), "theta" (decay center), "beta" (inverse softmax temperature), "mu0" (anticipated initial mean of all 4 options), "sigma0" (anticipated initial sd (uncertainty factor) of all 4 options), "sigmaD" (sd of diffusion noise)
#' @templateVar LENGTH_DATA_COLUMNS 3
#' @templateVar DETAILS_DATA_1 \item{"subjID"}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{"choice"}{Character representing the option chosen on the given trial: ( low == 1, high == 2 )}
#' @templateVar DETAILS_DATA_3 \item{"outcome"}{Integer value representing the outcome of the given trial: ( loss == 1, win == 2 )}
#' @templateVar DETAILS_DATA_4 \item{"cue"}{ first card shown: 1, 2, 3, 4, 5, 6, 7, 8, 9 }
#'
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#'
#' @references
#' Daw, N. D., O'Doherty, J. P., Dayan, P., Seymour, B., & Dolan, R. J. (2006). Cortical substrates
#'   for exploratory decisions in humans. Nature, 441(7095), 876-879.

bandit4arm2_kalman_filter <- hBayesDM_model(
  task_name       = "riggedDeck",
  model_name      = "kalman_filter",  <-------------------  CHANGE
  data_columns    = c("subjid", "cue", "choice", "outcome"),
  parameters      = list("lambda" = c(0, 0.9, 1),
                         "theta"  = c(0, 50, 100),
                         "beta"   = c(0, 0.1, 1),
                         "mu0"    = c(0, 85, 100),
                         "sigma0" = c(0, 6, 15),
                         "sigmaD" = c(0, 3, 15)),
  regressors      = list("beta_mean"     = 2,
                         "beta_samsiz"   = 2,
                         "weight_par"    = 2,
                         "weighted_mean" = 2),
  preprocess_func = function(raw_data, general_info) {
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj
    t_subjs <- general_info$t_subjs
    t_max   <- general_info$t_max
    
    choice  <- array(0, c(n_subj, t_max))
    outcome <- array(0, c(n_subj, t_max))
    cue     <- array(0, c(n_subj, t_max))

    for (i in 1:n_subj) {
      subj <- subjs[i]
      t <- t_subjs[i]
      DT_subj <- raw_data[subjid == subj]
      
      choice[i, 1:t]  <- DT_subj$choice
      outcome[i, 1:t] <- DT_subj$outcome
      cue[i, 1:t]     <- DT_subj$cue
    }
    
    data_list <- list(
      N       = n_subj,
      T       = t_max,
      Tsubj   = t_subjs,
      choice  = choice,
      outcome = outcome,
      cue     = cue
    )
    
    return(data_list)
  }
)
