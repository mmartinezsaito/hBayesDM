#' @templateVar MODEL_FUNCTION choiceRT_2dsd
#' @templateVar TASK_NAME Choice Reaction Time Task
#' @templateVar MODEL_NAME Two-Stage Dynamical Signal Detection Model
#' @templateVar MODEL_CITE (Pleskac & Busemeyer, 2010, Psychological Review)\cr *Note that this implementation is \strong{not} the 
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "choice", "RT", "confidence", "CT"
#' @templateVar PARAMETERS "alpha" (choice boundary separation), "beta" (choice bias), "delta" (pre-choice drift rate), "tau" (non-decision time), "alphax" (confidence level x boundary), "jump" (leap in accumulated evidence upon choice), "delta2" (post-choice drift rate)
#' @templateVar IS_NULL_POSTPREDS TRUE
#' @templateVar ADDITIONAL_ARG \code{RTbound}: Floating point value representing the lower bound (i.e., minimum allowed) reaction time. Defaults to 0.1 (100 milliseconds).
#' @templateVar LENGTH_DATA_COLUMNS 5
#' @templateVar DETAILS_DATA_1 \item{"subjID"}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{"choice"}{Choice made for the current trial, coded as \code{1}/\code{2} to indicate lower/upper boundary or left/right choices (e.g., 1 1 1 2 1 2).}
#' @templateVar DETAILS_DATA_3 \item{"RT"}{Choice reaction time for the current trial, in \strong{seconds} (e.g., 0.435 0.383 0.314 0.309, etc.).}
#' @templateVar DETAILS_DATA_4 \item{"confidence"}{Confidence judgment for the current trial, coded as \code{1}/\code{2}\code{3}\code{4} (e.g., 2 2 1 4 3 4, etc.).}
#' @templateVar DETAILS_DATA_5 \item{"CT"}{Confidence reaction time for the current trial, in \strong{seconds} (e.g., 0.435 0.383 0.314 0.309, etc.).}
#'
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#' @importFrom stats aggregate
#'
#' @description
#' Code for this model is based on choiceRT_ddm.R
#'
#' Parameters of the DDM (parameter names in Ratcliff), from \url{https://github.com/gbiele/stan_wiener_test/blob/master/stan_wiener_test.R}
#' \cr - alpha (a): Boundary separation or Speed-accuracy trade-off (high alpha means high accuracy). 0 < alpha
#' \cr - beta (b): Initial bias, for either response (beta > 0.5 means bias towards "upper" response 'A'). 0 < beta < 1
#' \cr - delta (v): Drift rate; Quality of the stimulus (delta close to 0 means ambiguous stimulus or weak ability). 0 < delta
#' \cr - tau (ter): Non-decision time + Motor response time + encoding time (high means slow encoding, execution). 0 < tau (in seconds)
#' \cr - alphax (ax): Non-decision time + Motor response time + encoding time (high means slow encoding, execution). 0 < tau (in seconds)
#' \cr - jump (j): Non-decision time + Motor response time + encoding time (high means slow encoding, execution). 0 < tau (in seconds)
#' \cr - delta2 (d2): Non-decision time + Motor response time + encoding time (high means slow encoding, execution). 0 < tau (in seconds)
#' NOTE: This model assumes symmetric decision and confidence thresholds, and evenly spaced confidence levels.
#'
#' @references
#' Pleskac TJ & Busemeyer JR (2010) Two-Stage Dynamic Signal Detection: A Theory of Choice, Decision Time, and Confidence. Psychological Review 117:864-901


choiceRT_2dsd <- hBayesDM_model(
  task_name       = "choiceRT",
  model_name      = "2dsd",
  data_columns    = c("subjID", "choice", "RT", "confidence", "CT"),
  parameters      = list("alpha"  = c(0, 0.5, Inf),
                         "beta"   = c(0, 0.5, 1),
                         "delta"  = c(0, 0.5, Inf),
                         "tau"    = c(0, 0.15, 1),
                         "alpha1" = c(0, 0.5, Inf),
                         "cr_unit"= c(0, 0.5, Inf),
                         "jump"   = c(0, 0.5, 1),
                         "delta2" = c(0, 0.5, Inf),
                         "tau2"   = c(0, 0.15, 1)),
  postpreds       = NULL,
  preprocess_func = function(raw_data, general_info, RTbound = 0.1) {
    # Use raw_data as a data.frame
    raw_data <- as.data.frame(raw_data)

    # Use general_info of raw_data
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj

    # Number of upper and lower boundary responses
    Nu <- with(raw_data, aggregate(choice == 2, by = list(y = subjid), FUN = sum)[["x"]])
    Nl <- with(raw_data, aggregate(choice == 1, by = list(y = subjid), FUN = sum)[["x"]])
    
    NC <- xtabs( ~subjid + choice + confidence, data=raw_data)

    # Reaction times for upper and lower boundary responses
    RTu <- array(-1, c(n_subj, max(Nu)))
    RTl <- array(-1, c(n_subj, max(Nl)))

    RTC <- array(-1, c(n_subj, max(NC), 2, 4))

    for (i in 1:n_subj) {
      subj <- subjs[i]
      subj_data <- subset(raw_data, raw_data$subjid == subj)

      RTu[i, 1:Nu[i]] <- subj_data$rt[subj_data$choice == 2]  # (Nu/Nl[i]+1):Nu/Nl_max will be padded with 0's
      RTl[i, 1:Nl[i]] <- subj_data$rt[subj_data$choice == 1]  # 0 padding is skipped in likelihood calculation
      
      for (ch in 1:2) 
        for (co in 1:4) 
          if (NC[i, ch, co] != 0)		 
            RTC[i, 1:NC[i, ch, co], ch, co] <- with(subj_data, ct[choice == ch & confidence == co]) 
    }

    # Minimum reaction time
    minRT <- with(raw_data, aggregate(rt, by = list(y = subjid), FUN = min)[["x"]])
    minCT <- aggregate(ct ~ subjid, FUN = min, data = raw_data)[["ct"]]

    # Wrap into a list for Stan
    data_list <- list(
      N       = n_subj,   # Number of subjects
      Nu_max  = max(Nu),  # Max (across subjects) number of upper boundary responses
      Nl_max  = max(Nl),  # Max (across subjects) number of lower boundary responses
      Nu      = Nu,       # Number of upper boundary responses for each subject
      Nl      = Nl,       # Number of lower boundary responses for each subject
      NC_max  = max(NC), 
      Nu1     = NC[,2,1],
      Nu2     = NC[,2,2],
      Nu3     = NC[,2,3],
      Nu4     = NC[,2,4],
      Nl1     = NC[,1,1],
      Nl2     = NC[,1,2],
      Nl3     = NC[,1,3],
      Nl4     = NC[,1,4],
      RTu     = RTu,      # Upper boundary response times
      RTl     = RTl,      # Lower boundary response times
      RTu1    = RTC[,,2,1],
      RTu2    = RTC[,,2,2],
      RTu3    = RTC[,,2,3],
      RTu4    = RTC[,,2,4],
      RTl1    = RTC[,,1,1],
      RTl2    = RTC[,,1,2],
      RTl3    = RTC[,,1,3],
      RTl4    = RTC[,,1,4],
      minRT   = minRT,    # Minimum RT for each subject
      minCT   = minCT,
      RTbound = RTbound   # Lower bound of RT across all subjects (e.g., 0.1 second)
    )

    # Returned data_list will directly be passed to Stan
    return(data_list)
  }
)

