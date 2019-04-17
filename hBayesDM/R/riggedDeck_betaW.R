#' @templateVar MODEL_FUNCTION betaW
#' @templateVar CONTRIBUTOR Mario Martinez-Saito
#' @templateVar TASK_NAME Rigged card deck
#' @templateVar MODEL_NAME Weighted Bayesian Beta learner  
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "cue", "choice", "outcome"
#' @templateVar PARAMETERS "Bnu0" (initial Beta distribution sample size), "sw" (weight) 
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

riggedDeck_betaW <- hBayesDM_model(
  task_name       = "riggedDeck",
  model_name      = "betaW", 
  data_columns    = c("subjid", "cue", "choice", "outcome"),
  parameters      = list("Bnu0" = c(0, 10, inf),
                         "sw"   = c(0, 0.5, 1)),
  regressors      = list("beta_mean"     = 2,
                         "beta_samsiz"   = 2,
                         "weighted_mean" = 2),
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
