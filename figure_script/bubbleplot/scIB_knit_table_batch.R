

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
    data,
    ranked_data,
    column_info,
    row_info,
    na_idx,
    isImage = TRUE
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
  #data <- data[order(data$overall, decreasing = T), ]
  # gather circle data
  ind_circle <- which(column_info$geom == "circle")
  if(length(ind_circle) > 0){
    dat_mat_value <- as.matrix(data[, ind_circle, drop = FALSE])
    dat_mat_rank <- as.matrix(ranked_data[, names(data[, ind_circle, drop = FALSE]), drop = FALSE])
    
    col_palette <- data.frame(metric = colnames(dat_mat_value), 
                              group = column_info[match(colnames(dat_mat_value), column_info$id), "geom"])
    
    circle_data <- data.frame(
      label = unlist(lapply(colnames(dat_mat_rank), function(x) rep(x, nrow(dat_mat_rank)))),  # Repeat column names
      x0 = unlist(lapply(column_pos$x[ind_circle], function(x) rep(x, nrow(dat_mat_rank)))),  # Repeat x positions
      y0 = rep(row_pos$y, ncol(dat_mat_rank)),  # Repeat y positions
      r = row_height / 2 * as.vector(sqrt(dat_mat_rank)),  # Circle radius
      rownames = rep(rownames(dat_mat_value), times = ncol(dat_mat_rank))  # Add rownames of dat_mat
    )
    
    for(l in unique(circle_data$label)){
      ind_l <- which(circle_data$label == l)
      circle_data[ind_l, "r"] <- rescale(circle_data[ind_l, "r"], to = c(0.15, 0.85), from = c(0,1))
      #circle_data[ind_l, "r"] <- rescale(circle_data[ind_l, "r"], to = c(0.15, 0.55), from = range(circle_data[ind_l, "r"], na.rm = T))
    }
    
    #for (i in 1:nrow(na_idx)) {
    #  circle_data$r[which(circle_data$label == na_idx[i, 2] & circle_data$rownames == na_idx[i, 1])] <- 0.05
    #}
    
    palette <- colorRampPalette(brewer.pal(9, "Greens"))
    all_colour <- palette(200)  
    color_index <- round(dat_mat_value * (length(all_colour) - 1)) + 1 #rank(unlist(dat_mat_value[,1:6]), ties.method = "max") #
    circle_data$colors <- all_colour[color_index]
  }
  
  
  # gather bar data
  ind_bar <- which(column_info$geom == "bar")
  dat_mat_value <- as.matrix(data[, ind_bar]) # not ranked
  dat_mat_rank <- as.matrix(ranked_data[, "overall"])
  colnames(dat_mat_value) <- "overall"
  colnames(dat_mat_rank) <- "overall"

  rect_data <- data.frame(label = unlist(lapply(colnames(dat_mat_value),
                                                function(x) rep(x, nrow(dat_mat_value)))),
                          method = rep(row_info$id, ncol(dat_mat_value)),
                          value = as.vector(dat_mat_rank), 
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
      xmin = xmin + (1 - value) * xwidth * hjust,
      xmax = xmax - (1 - value) * xwidth * (1 - hjust)
    )

  colors <- NULL
  palette <- colorRampPalette(brewer.pal(9, palettes[["batch.overall"]]))
  all_colour <- palette(200)  
  color_index <- round(dat_mat_value * (length(all_colour) - 1)) + 1
  colors <- c(colors, all_colour[color_index])
  
  
  
  rect_data$colors <- colors
  
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
  
  # ADD COLUMN NAMES
  df <- column_pos %>% filter(id != "method")
  df2 <- data.frame(id = df$id, x = NA)
  
  df2[which(df2$id %in% intersect(df$id, circle_data$label)), "x"] <- 
    unique(circle_data[which(circle_data$label %in% intersect(df$id, circle_data$label)), "x0"]) + 3
  
  df2[which(df2$id %in% intersect(df$id, rect_data$label)), "x"] <- 
    na.omit(unique(rect_data[which(rect_data$label %in% intersect(df$id, rect_data$label)), "xmin"])) + 3
  
  df2[which(df2$id %in% intersect(df$id, text_data$group)), "x"] <- 
    unique(text_data[which(text_data$group %in% intersect(df$id, text_data$group)), "xmin"]) + 2
  
  df2$x[2] <- 3.5 + 3
  if (!is.na(df2$x[3])){df2$x[3] <- 3.5 + 4.2}
  
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
  
  # gather image data
  ind_img <- which(column_info$geom == "image")
  if(length(ind_img) > 0){
    dat_mat <- as.matrix(data[, ind_img])
    
    image_data <- data.frame(x = unlist(lapply(column_pos$x[ind_img[1]], 
                                               function(x) rep(x, nrow(dat_mat)))), 
                             y = rep(row_pos$y),
                             image = mapvalues(dat_mat, from = c("Yes", "No", "Python", "Python/R", "R", "R/Python"), 
                                               to = c("Di/code/icons/yes.png", "Di/code/icons/no.png", "Di/code/icons/python.png", 
                                                      "Di/code/icons/python_R.png", "Di/code/icons/R.png", "Di/code/icons/python_R.png")),
                             stringsAsFactors = FALSE
    )
  }
  
  
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
  
  ####################################
  ###   CREATE HARDCODED LEGENDS   ###
  ####################################
  
  x_min_score <-  minimum_x + 10
  
  leg_max_y <- minimum_y - .5
  
  
  # CREATE LEGEND for ranking colors
  leg_min_x <- 5
  rank_groups <- c("batch.overall", "batch.metric")
  leg_max_x <- leg_min_x+3
  
  rank_title_data <- data.frame(xmin = leg_min_x + 3, 
                                xmax = leg_min_x + 3, 
                                ymin = leg_max_y - 1, 
                                ymax = leg_max_y, 
                                label_value = "Score", 
                                hjust = 0, vjust = 0, 
                                fontface = "bold")
  
  rank_minimum_x <- list("batch.overall" = leg_min_x, 
                         "batch.metric" = leg_min_x + 1)
  
  for(rg in rank_groups){
    rank_palette <- colorRampPalette((brewer.pal(9, palettes[[rg]])))(41)
    
    rank_data <- data.frame(xmin = rank_minimum_x[[rg]],
                            xmax = rank_minimum_x[[rg]] + .8,
                            ymin = seq(leg_max_y - 4, leg_max_y - 2, by = .05),
                            ymax = seq(leg_max_y - 3.5, leg_max_y - 1.5, by = .05),
                            border = FALSE,
                            colors = rank_palette
    )
    
    rank_value_data <- data.frame(xmin = rank_data$xmin + 0.8,
                                  xmax = rank_data$xmax + 0.8,
                                  ymin = rank_data$ymin,
                                  ymax = rank_data$ymax,
                                  hjust = 0, vjust = 0, size = 2.5,
                                  label_value = "")
    
    rank_value_data$label_value[which.max(rank_value_data$ymax)] <- "High"
    rank_value_data$label_value[which.min(rank_value_data$ymax)] <- "Low"
    
    rect_data <- bind_rows(rect_data, rank_data)
  }
  
  text_data <- bind_rows(text_data, rank_title_data, rank_value_data)
  
  # CREATE LEGEND for circle scores
  # circle legend
  
  cir_minimum_x <- x_min_score
  
  cir_legend_size <- 1
  cir_legend_space <- .1
  
  if (nrow(data) > 5) {
    cir_legend_dat <-
      data.frame(
        value = seq(0, 1, by = .2),
        r = row_height/2*seq(0, 1, by = .2)
      )
  } else {
    cir_legend_dat <-
      data.frame(
        value = seq(0, 1, length.out = 3),
        r = row_height/2*seq(0, 1, length.out = 3)
      )
  }
  
  cir_legend_dat$r <- rescale(cir_legend_dat$r, to = c(0.15, 0.55), from = range(cir_legend_dat$r, na.rm = T))
  
  x0 <- vector("integer", nrow(cir_legend_dat))
  for(i in 1:length(x0)){
    if(i == 1){
      x0[i] <- cir_minimum_x + cir_legend_space + cir_legend_dat$r[i]
    }
    else {
      x0[i] <- x0[i-1] + cir_legend_dat$r[i-1] + cir_legend_space + cir_legend_dat$r[i]
    }
  }
  
  cir_legend_dat$x0 <- x0
  cir_legend_min_y <- leg_max_y - 4
  cir_legend_dat$y0 <- cir_legend_min_y + 1 + cir_legend_dat$r
  
  cir_legend_dat$colors <- NULL
  cir_maximum_x <- max(cir_legend_dat$x0)
  
  ## circle title ("Rank")
  cir_title_data <- data_frame(xmin = cir_minimum_x + 3, 
                               xmax = cir_maximum_x + 3, 
                               ymin = leg_max_y -1, 
                               ymax = leg_max_y,
                               label_value = "Rank", 
                               hjust = 0, vjust = 0, fontface = "bold")
  
  cir_value_data <- data.frame(xmin = cir_legend_dat$x0 - cir_legend_dat$r + 3,
                               xmax = cir_legend_dat$x0 + cir_legend_dat$r + 3,
                               ymin = cir_legend_min_y,
                               ymax = cir_legend_min_y + 3,
                               hjust = 0, vjust = 0, size = 2.5,
                               label_value = ifelse(cir_legend_dat$value == 0, as.character(nrow(data)),
                                                    ifelse(cir_legend_dat$value == 1, "1", "")))
  
  cir_value_data$hjust[which(cir_value_data$label_value == 1)] <- cir_value_data$hjust[which(cir_value_data$label_value == 1)] + 0.5
  
  circle_data <- bind_rows(circle_data, cir_legend_dat)
  text_data <- bind_rows(text_data, cir_title_data, cir_value_data)
  
  minimum_y <- min(minimum_y, min(text_data$ymin, na.rm = TRUE))
  
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
    g <- g + geom_rect(aes(xmin = min(column_pos$xmin) - 2, xmax = max(column_pos$xmax) + 5, 
                           ymin = ymin - (row_space / 2), ymax = ymax + (row_space / 2)), df, fill = "#DDDDDD")
  } 
  
  # PLOT CIRCLES
  circle_data <- circle_data[!is.na(circle_data$label),]
  if (length(ind_circle) > 0) {
    g <- g + ggforce::geom_circle(aes(x0 = x0 + 3, y0 = y0, fill= colors, r = r), circle_data[!is.na(circle_data$label),], size=.25)
  }
  
  
  # PLOT RECTANGLES
  rect_data <- rect_data[!is.na(rect_data$label),]
  if (nrow(rect_data) > 0) {
    # add defaults for optional values
    rect_data <- rect_data %>%
      add_column_if_missing(alpha = 1, border = TRUE, border_colour = "black") %>%
      mutate(border_colour = ifelse(border, border_colour, NA))
    
    g <- g + geom_rect(aes(xmin = xmin + 3, xmax = xmax + 3, ymin = ymin, ymax = ymax, 
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

    text_data[text_data$group == "programmingLaguage", "x"] <- text_data[text_data$group == "programmingLaguage", "x"] + 3
    text_data[text_data$group == "DL", "x"] <- text_data[text_data$group == "DL", "x"] + 5
    text_data[text_data$group == "celltype", "x"] <- text_data[text_data$group == "celltype", "x"] + 2
    
    text_data <- text_data[!is.na(text_data$label),]
    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, hjust = hjust, 
                           vjust = vjust, size = size, fontface = fontface, angle = angle), 
                       data = text_data)
    
    text_data_left[text_data_left$group == "Method", "x"] <- text_data_left[text_data_left$group == "Method", "x"]
    
    text_data_left$size[which(text_data_left$label_value %in% colnames(data))] <- 2.5
    
    idx <- which(text_data_left$label_value %in% c("celltype", "programmingLaguage", "overall", "DL"))
    text_data_left$x[idx] <- text_data_left$x[idx] + 0.5
    
    text_data_left$label_value <- ifelse(is.na(match(text_data_left$label_value, label_match$label_old)), 
                                         text_data_left$label_value, 
                                         label_match$label_new[match(text_data_left$label_value, label_match$label_old)])
    
    text_data_left <- text_data_left[1:(dim(text_data_left)[1]-6),]
    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, 
                           hjust = hjust, vjust = vjust, size = size, 
                           fontface = fontface, angle = angle), 
                       data = text_data_left)
  }
  
  # PLOT IMAGES
  
  if (isImage) {
    for(r in 1:nrow(image_data)){
      g <- g + cowplot::draw_image(image = image_data$image.programmingLaguage[r],
                                   x = image_data[r, "x"] + 2,
                                   y = image_data[r, "y"]-.5)
      g <- g + cowplot::draw_image(image = image_data$image.DL[r],
                                   x = image_data[r, "x"] + 3.5,
                                   y = image_data[r, "y"]-.5)
    }
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


