#' @templateVar MODEL_FUNCTION dd_cs
#' @templateVar TASK_NAME Delay Discounting Task
#' @templateVar MODEL_NAME Constant-Sensitivity (CS) Model
#' @templateVar MODEL_CITE (Ebert & Prelec, 2007, Management Science)
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "delay_later", "amount_later", "delay_sooner", "amount_sooner", "choice"
#' @templateVar PARAMETERS "r" (exponential discounting rate), "s" (impatience), "beta" (inverse temperature)
#' @templateVar LENGTH_DATA_COLUMNS 6
#' @templateVar DETAILS_DATA_1 \item{"subjID"}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{"delay_later"}{An integer representing the delayed days for the later option (e.g. 1, 6, 28).}
#' @templateVar DETAILS_DATA_3 \item{"amount_later"}{A floating point number representing the amount for the later option (e.g. 10.5, 13.4, 30.9).}
#' @templateVar DETAILS_DATA_4 \item{"delay_sooner"}{An integer representing the delayed days for the sooner option (e.g. 0).}
#' @templateVar DETAILS_DATA_5 \item{"amount_sooner"}{A floating point number representing the amount for the sooner option (e.g. 10).}
#' @templateVar DETAILS_DATA_6 \item{"choice"}{If amount_later was selected, choice == 1; else if amount_sooner was selected, choice == 0.}
#'
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#'
#' @references
#' Ebert, J. E. J., & Prelec, D. (2007). The Fragility of Time: Time-Insensitivity and Valuation of
#'   the Near and Far Future. Management Science. http://doi.org/10.1287/mnsc.1060.0671

dd_cs <- hBayesDM_model(
  task_name       = "dd",
  model_name      = "cs",
  data_columns    = c("subjID", "delay_later", "amount_later", "delay_sooner", "amount_sooner", "choice"),
  parameters      = list("r"    = c(0, 0.1, 1),
                         "s"    = c(0, 1, 10),
                         "beta" = c(0, 1, 5)),
  preprocess_func = function(raw_data, general_info) {
    # Currently class(raw_data) == "data.table"

    # Use general_info of raw_data
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj
    t_subjs <- general_info$t_subjs
    t_max   <- general_info$t_max

    # Initialize (model-specific) data arrays
    delay_later   <- array( 0, c(n_subj, t_max))
    amount_later  <- array( 0, c(n_subj, t_max))
    delay_sooner  <- array( 0, c(n_subj, t_max))
    amount_sooner <- array( 0, c(n_subj, t_max))
    choice        <- array(-1, c(n_subj, t_max))

    # Write from raw_data to the data arrays
    for (i in 1:n_subj) {
      subj <- subjs[i]
      t <- t_subjs[i]
      DT_subj <- raw_data[subjid == subj]

      delay_later[i, 1:t]   <- DT_subj$delaylater
      amount_later[i, 1:t]  <- DT_subj$amountlater
      delay_sooner[i, 1:t]  <- DT_subj$delaysooner
      amount_sooner[i, 1:t] <- DT_subj$amountsooner
      choice[i, 1:t]        <- DT_subj$choice
    }

    # Wrap into a list for Stan
    data_list <- list(
      N             = n_subj,
      T             = t_max,
      Tsubj         = t_subjs,
      delay_later   = delay_later,
      amount_later  = amount_later,
      delay_sooner  = delay_sooner,
      amount_sooner = amount_sooner,
      choice        = choice
    )

    # Returned data_list will directly be passed to Stan
    return(data_list)
  }
)

