
library(scales)
library(ggimage)
library(cowplot)

add_column_if_missing <- function(df, ...) {
  column_values <- list(...)
  for (column_name in names(column_values)) {
    default_val <- rep(column_values[[column_name]], nrow(df))
    
    if (column_name %in% colnames(df)) {
      df[[column_name]] <- ifelse(is.na(df[[column_name]]), default_val, df[[column_name]])
    } else {
      df[[column_name]] <- default_val
    }
  }
  df
}

scIB_knit_table <- function(
    data, ranked_data, 
    column_info,
    row_info, na_idx = na_idx, isImage = FALSE
) {
  # no point in making these into parameters
  row_height <- 1.1
  row_space <- .1
  row_bigspace <- .5
  col_width <- 1.1
  col_space <- .2
  col_bigspace <- .5
  segment_data <- NULL
  
  # DETERMINE ROW POSITIONS
  row_info$group <- ""
  row_pos <-
    row_info %>%
    group_by(group) %>%
    dplyr::mutate(group_i = row_number()) %>%
    ungroup() %>%
    dplyr::mutate(
      row_i = row_number(),
      colour_background = group_i %% 2 == 1,
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      ysep = ifelse(do_spacing, row_height + 2 * row_space, row_space),
      y = - (row_i * row_height + cumsum(ysep)),
      ymin = y - row_height / 2,
      ymax = y + row_height / 2
    )
  
  # DETERMINE COLUMN POSITIONS
  column_info$group <- ""
  
  column_info <-
    column_info %>%
    add_column_if_missing(width = col_width, overlay = FALSE)
  
  column_pos <-
    column_info %>%
    mutate(
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      xsep = case_when(
        overlay ~ c(0, -head(width, -1)),
        do_spacing ~ col_bigspace,
        TRUE ~ col_space
      ),
      xwidth = case_when(
        overlay & width < 0 ~ width - xsep,
        overlay ~ -xsep,
        TRUE ~ width
      ),
      xmax = cumsum(xwidth + xsep),
      xmin = xmax - xwidth,
      x = xmin + xwidth / 2
    )
  
  ##########################
  #### CREATE GEOM DATA ####
  ##########################
  
  # gather bar data
  ind_bar <- which(column_info$geom == "bar")
  dat_mat_value <- as.matrix(data[, ind_bar, drop = FALSE])
  dat_mat_rank <- as.matrix(ranked_data[, names(data[, ind_bar, drop = FALSE])])
  
  
  rect_data <- data.frame(label = unlist(lapply(colnames(dat_mat_value),
                                                function(x) rep(x, nrow(dat_mat_value)))),
                          method = rep(data$method, ncol(dat_mat_value)),
                          value_rank = as.vector(dat_mat_rank),
                          value = as.vector(dat_mat_value),
                          xmin = unlist(lapply(column_pos[ind_bar, "xmin"],
                                               function(x) rep(x, nrow(dat_mat_value)))),
                          xmax = unlist(lapply(column_pos[ind_bar, "xmax"],
                                               function(x) rep(x, nrow(dat_mat_value)))),
                          ymin = rep(row_pos$ymin, ncol(dat_mat_value)),
                          ymax = rep(row_pos$ymax, ncol(dat_mat_value)),
                          xwidth = unlist(lapply(column_pos[ind_bar, "xwidth"],
                                                 function(x) rep(x, nrow(dat_mat_value))))
  )
  rect_data <- rect_data %>%
    add_column_if_missing(hjust = 0) %>%
    mutate(
      xmin = xmin + (1 - value_rank) * xwidth * hjust,
      xmax = xmax - (1 - value_rank) * xwidth * (1 - hjust)
    )
  
  
  palette <- colorRampPalette(brewer.pal(9, "Blues"))
  all_colour <- palette(200) 
  color_index <- round(dat_mat_value * (length(all_colour) - 1)) + 1
  rect_data$colors <- all_colour[color_index]
  
  # gather text data
  ind_text <- which(column_info$geom == "text")
  dat_mat <- as.matrix(data[, ind_text])
  colnames(dat_mat)[1] <- "Method"
  
  text_data <- data.frame(label_value = as.vector(dat_mat), 
                          group = rep(colnames(dat_mat), each = nrow(dat_mat)),
                          xmin = unlist(lapply(column_pos[ind_text, "xmin"], 
                                               function(x) rep(x, nrow(dat_mat)))),
                          xmax = unlist(lapply(column_pos[ind_text, "xmax"], 
                                               function(x) rep(x, nrow(dat_mat)))),
                          ymin = rep(row_pos$ymin, ncol(dat_mat)),
                          ymax = rep(row_pos$ymax, ncol(dat_mat)),
                          size = 3, fontface = "plain", stringsAsFactors = F)
  
  text_data$colors <- "black"
  
  suppressWarnings({
    minimum_x <- min(column_pos$xmin, segment_data$x, segment_data$xend, 
                     text_data$xmin, na.rm = TRUE)
    maximum_x <- max(column_pos$xmax, segment_data$x, segment_data$xend, 
                     text_data$xmax, na.rm = TRUE)
    minimum_y <- min(row_pos$ymin, segment_data$y, segment_data$yend,  
                     text_data$ymin, na.rm = TRUE)
    maximum_y <- max(row_pos$ymax, segment_data$y, segment_data$yend, 
                     text_data$ymax, na.rm = TRUE)
  })
  
  # ADD COLUMN NAMES
  df <- column_pos %>% filter(id != "method")
  df2 <- data.frame(id = df$id, x = NA)
  
  df2[which(df2$id %in% intersect(df$id, rect_data$label)), "x"] <- 
    na.omit(unique(rect_data[which(rect_data$label %in% intersect(df$id, rect_data$label)), "xmin"])) + 3.5
  
  df2[which(df2$id %in% intersect(df$id, text_data$group)), "x"] <- 
    unique(text_data[which(text_data$group %in% intersect(df$id, text_data$group)), "xmin"]) + 2
  
  if (nrow(df) > 0) {
    segment_data <- segment_data %>% bind_rows(
      df %>% transmute(x = x, xend = x, y = -.3, yend = -.1, size = .5)
    )
    text_data <-
      bind_rows(
        text_data,
        df2 %>% transmute(
          xmin = x, xmax = x, ymin = 0, ymax = -0.5,
          angle = 30, vjust = 0, hjust = 0,
          label_value = id, 
          size = 3
        )
      )
  }
  
  ####################################
  ###   CREATE HARDCODED LEGENDS   ###
  ####################################
  
  x_min_score <-  minimum_x + 10
  
  leg_max_y <- minimum_y - .5
  
  
  # CREATE LEGEND for ranking colors
  leg_min_x <- 5
  rank_groups <- c("clustering.overall", "clustering.metric")
  leg_max_x <- leg_min_x+3
  
  rank_title_data <- data.frame(xmin = leg_min_x + 2, 
                                xmax = leg_min_x + 2, 
                                ymin = leg_max_y - 1, 
                                ymax = leg_max_y, 
                                label_value = "Score", 
                                hjust = 0, vjust = 0, 
                                fontface = "bold")
  
  rank_minimum_x <- list("clustering.overall" = leg_min_x, 
                         "clustering.metric" = leg_min_x + 1)
  
  for(rg in rank_groups){
    rank_palette <- colorRampPalette((brewer.pal(9, palettes[[rg]])))(5)
    
    rank_data <- data.frame(xmin = rank_minimum_x[[rg]],
                            xmax = rank_minimum_x[[rg]] + .8,
                            ymin = seq(leg_max_y-4, leg_max_y - 2, by = .5),
                            ymax = seq(leg_max_y-3.5, leg_max_y -1.5, by = .5),
                            border = TRUE,
                            colors = rank_palette
    )
    
    rank_value_data <- data.frame(xmin = rank_data$xmin,
                                  xmax = rank_data$xmax,
                                  ymin = rank_data$ymin,
                                  ymax = rank_data$ymax,
                                  hjust = 0, vjust = 0, size = 2.5,
                                  label_value = "")
    rank_value_data$label_value[which.max(rank_value_data$ymax)] <- "High"
    rank_value_data$label_value[which.min(rank_value_data$ymax)] <- "Low"
    
    rect_data <- bind_rows(rect_data, rank_data)
  }
  
  text_data <- bind_rows(text_data, rank_title_data, rank_value_data)
  
  ########################
  ##### COMPOSE PLOT #####
  ########################
  
  g <-
    ggplot() +
    coord_equal(expand = TRUE) +
    scale_alpha_identity() +
    scale_colour_identity() +
    scale_fill_identity() +
    scale_size_identity() +
    scale_linetype_identity() +
    cowplot::theme_nothing()
  
  # PLOT ROW BACKGROUNDS
  df <- row_pos %>% dplyr::filter(colour_background)
  if (nrow(df) > 0) {
    g <- g + geom_rect(aes(xmin = min(column_pos$xmin) - 2, xmax = max(column_pos$xmax) + 2.5, 
                           ymin = ymin - (row_space / 2), ymax = ymax + (row_space / 2)), df, fill = "#DDDDDD")
  } 
  
  # PLOT RECTANGLES
  if (nrow(rect_data) > 0) {
    # add defaults for optional values
    rect_data <- rect_data %>%
      add_column_if_missing(alpha = 1, border = TRUE, border_colour = "black") %>%
      mutate(border_colour = ifelse(border, border_colour, NA))
    
    g <- g + geom_rect(aes(xmin = xmin + 2, xmax = xmax + 2, ymin = ymin, ymax = ymax, 
                           fill = colors, colour = border_colour, alpha = alpha), 
                       rect_data, size = .25)
  }
  
  
  # PLOT TEXT
  if (nrow(text_data) > 0) {
    # add defaults for optional values
    text_data <- text_data %>%
      add_column_if_missing(
        hjust = .5,
        vjust = .5,
        size = 3,
        fontface = "plain",
        colors = "black",
        lineheight = 1,
        angle = 0
      ) %>%
      mutate(
        angle2 = angle / 360 * 2 * pi,
        cosa = cos(angle2) %>% round(2),
        sina = sin(angle2) %>% round(2),
        alphax = ifelse(cosa < 0, 1 - hjust, hjust) * abs(cosa) + ifelse(sina > 0, 1 - vjust, vjust) * abs(sina),
        alphay = ifelse(sina < 0, 1 - hjust, hjust) * abs(sina) + ifelse(cosa < 0, 1 - vjust, vjust) * abs(cosa),
        x = (1 - alphax) * xmin + alphax * xmax,
        y = (1 - alphay) * ymin + alphay * ymax
      ) %>%
      filter(label_value != "")
    
    # subset text_data to left-aligned rows
    text_data$group[which(is.na(text_data$group))] <- "Method"
    text_data_left <- text_data[which(text_data$group == "Method"), ]
    text_data <- text_data[-which(text_data$group == "Method"), ]
    
    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, hjust = hjust, 
                           vjust = vjust, size = size, fontface = fontface, angle = angle), 
                       data = text_data)
    
    text_data_left[text_data_left$group == "Method", "x"] <- text_data_left[text_data_left$group == "Method", "x"]
    # change the position of column names
    text_data_left$x[which(text_data_left$label_value %in% colnames(data))] <- text_data_left$x[which(text_data_left$label_value %in% colnames(data))] - 1
    text_data_left$y[which(text_data_left$label_value %in% colnames(data))] <- text_data_left$y[which(text_data_left$label_value %in% colnames(data))] - 0.5
    
    text_data_left$label_value <- gsub("\\.", " ", text_data_left$label_value)
    
    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, 
                           hjust = hjust, vjust = vjust, size = size, 
                           fontface = fontface, angle = angle), 
                       data = text_data_left)
  }
  
  # ADD SIZE
  # reserve a bit more room for text that wants to go outside the frame
  minimum_x <- minimum_x - 4
  maximum_x <- maximum_x + 2
  minimum_y <- minimum_y - 2
  maximum_y <- maximum_y + 4
  
  g$width <- maximum_x - minimum_x
  
  g <- g + expand_limits(x = c(minimum_x, maximum_x), y = c(minimum_y, maximum_y))
  
  return(g)
  
}


