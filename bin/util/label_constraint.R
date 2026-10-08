# R copy of label_key and constrain from label_constraint.py (see there), for the R wrappers.

label_key <- function(s) tolower(trimws(gsub("[[:space:]]+", " ", s)))

constrain <- function(labels, path, method, truth) {
  table <- read.delim(path, colClasses = "character", na.strings = character(), encoding = "UTF-8")
  listed <- setNames(table$listed, table$label)[label_key(labels)]
  kept <- setNames(listed[!is.na(listed)], labels[!is.na(listed)])
  missing <- sort(setdiff(unique(truth), kept))
  if (length(missing))
    message(sprintf("WARNING %s (%s): %d of %d listed cell types have no matching label: %s",
                    method, basename(path), length(missing), length(unique(truth)),
                    paste(missing, collapse = "; ")))
  kept
}
