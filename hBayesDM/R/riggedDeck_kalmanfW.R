#' @templateVar MODEL_FUNCTION kalmanfW
#' @templateVar CONTRIBUTOR Mario Martinez-Saito
#' @templateVar TASK_NAME Rigged deck game
#' @templateVar MODEL_NAME Weighted Kalman filter
#' @templateVar MODEL_CITE (Daw et al., 2006, Nature)
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "cue", "choice", "outcome"
#' @templateVar PARAMETERS "sigma0" (initial stdev of winning probabilities of all cards), "sigmaO" (observation noise), "sigmaE" (evolution noise), "w" (weight between Kalma filter and default model)
#' @templateVar LENGTH_DATA_COLUMNS 4
#' @templateVar DETAILS_DATA_1 \item{"subjID"}{ A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{"cue"}{ First card shown: 1, 2, 3, 4, 5, 6, 7, 8, 9 }
#' @templateVar DETAILS_DATA_3 \item{"choice"}{ Character representing the option chosen on the given trial: ( low == 0, high == 1 )}
#' @templateVar DETAILS_DATA_4 \item{"outcome"}{ Integer value representing the outcome of the given trial: ( loss == 0, win == 1 )}
#'
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#'
#' @references
#' Daw, N. D., O'Doherty, J. P., Dayan, P., Seymour, B., & Dolan, R. J. (2006). Cortical substrates
#'   for exploratory decisions in humans. Nature, 441(7095), 876-879.

riggedDeck_kalmanfW <- hBayesDM_model(
  task_name       = "riggedDeck",
  model_name      = "kalmanfW", 
  data_columns    = c("subjid", "cue", "choice", "outcome"),
  parameters      = list("sigma0" = c(0, 1, inf),
                         "sigmaO" = c(0, 1, inf),
                         "sigmaE" = c(0, 1, inf),
			 "w"      = c(0, .5, inf)),
  regressors      = list("winp_mean" = 2,
                         "winp_var"  = 2,
                         "gain"      = 2,
                         "rpe"       = 2),
  preprocess_func = function(raw_data, general_info) {
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj
    t_subjs <- general_info$t_subjs
    t_max   <- general_info$t_max
    
    choice  <- array(0, c(n_subj, t_max))
    outcome <- array(0, c(n_subj, t_max))
    cue     <- array(1, c(n_subj, t_max))

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
