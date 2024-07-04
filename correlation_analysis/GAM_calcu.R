install.packages('geojsonio')
install.packages('ggeffects')
install.packages('gratia')
install.packages('pracma')

writeLines('PATH="${RTOOLS40_HOME}\\usr\\bin;${PATH}"', con = "~/.Renviron")
Sys.which("make")

# devtools::install_github('astrzalka/findpeaks')
#library(findpeaks)
#findpeaks::run_app()

library(ggeffects)
library(gratia)
library(geojsonio)
library(sf)
library(mgcv)
library(dplyr)

gdp_scaled <- read.csv("D:/WUR/master/MLP/master thesis/data/数据汇总/PC6_corr_df_scaled.csv")

# -----------gam
formula <- WalkabilityIndex ~ s(totGVI,k=4)+s(VSD,k=4)+s(top,k=4)+s(low,k=4)+

# optimal gam (cross-validation)
optimized_gc_gam <- gam(formula, data = gdp_scaled, method = "GCV.Cp")
# Get the predictions from the GAM model
predictions_gc <- predict(optimized_gc_gam)
# Calculate the Mean Squared Error (MSE)
mse_gc <- round(mean((gdp_scaled$WalkabilityIndex - predictions_gc)^2),3)

gam_summary <- summary(optimized_gc_gam)
gam_summary

# method-rsml
optimized_rsml_gam <- gam(formula, data = gdp_scaled, select = TRUE, method = "REML")
predictions_rsml <- predict(optimized_rsml_gam)
mse_rsml <- mean((gdp_scaled$WalkabilityIndex - predictions_rsml)^2)
# method-ml
optimized_ml_gam <- gam(formula, data = gdp_scaled, select = TRUE, method = "ML")
predictions_ml <- predict(optimized_ml_gam)
mse_ml <- mean((gdp_scaled$WalkabilityIndex - predictions_ml)^2)

mgcv::concurvity(optimized_gc_gam)

round(summary(optimized_gc_gam)$r.sq,3) # adjusted R squared
round(AIC(optimized_gc_gam),3)
round(coef(optimized_gc_gam),3)

# plot整体partial effect plots
draw(optimized_gc_gam, residuals = FALSE)

# Set the figure size using the par() function
par(mfrow = c(1, 1), mar = c(4, 4, 2, 2))
# Use the draw function
draw(optimized_gc_gam, residuals = FALSE)

pdf("plot.pdf", width = 40, height = 30)  # Adjust width and height as needed
draw(optimized_gc_gam, residuals = FALSE,cex.main = 4, cex.lab = 4, cex.axis = 4, cex.sub = 8)
dev.off()


# Calculate quantiles (0.25, mean, 0.75) for each column
df <- gdp_scaled[, 3:ncol(gdp_scaled)]
quantiles_df <- data.frame(
  Column = colnames(df),
  Quantile_0.25 = sapply(df, function(x) quantile(x, 0.25)),
  Mean = sapply(df, mean),
  Quantile_0.75 = sapply(df, function(x) quantile(x, 0.75))
)
# Transpose the resulting data frame to have 5 rows and 3 columns
summary_df <- t(quantiles_df[, -1])

colnames(summary_df) <- c('totGVI','VSD','top','low',
                          'building_height','vitality_level','noise_level','density','width_index')


# Save the model summary to a text file (not CSV)
writeLines(capture.output(gam_summary), "D:/WUR/master/MLP/master thesis/data/数据汇总/corr_analysis/GAM_summary.txt")

# ------------partial plot
# Calculate quartiles (or other quantiles) for the variable
quantiles <- round(quantile(gdp_scaled$totGVI, probs = c(0.25, 0.5, 0.75)),3)
# Print the calculated thresholds
print(quantiles)

# Create the partial plot
plot(optimized_gc_gam, select = 1,lwd = 2)
# Add vertical lines at the calculated thresholds
abline(v = quantiles, col = "red", lwd = 2)
# Add markers and labels below the lines
text(quantiles[1], -0.02, "25%", pos = 1, cex = 1.5, col = "red")
text(mean(quantiles), -0.08, "50%", pos = 1, cex = 1.5, col = "red")
text(quantiles[3], -0.02, "75%", pos = 1, cex = 1.5, col = "red")
# Add a title
title("Partial model of GAM - totGVI")
# Add a legend
legend("topright", legend = "quantile", col = "red", lwd = 2, cex = 1)

# -----------------------
Sys.which("make")
findpeaks::run_app()

library(tibble)
x_range <- seq(min(gdp_scaled$totGVI), max(gdp_scaled$totGVI), length = 496)
new_data <- data.frame(
  x_variable = x_range,
  low = seq(min(gdp_scaled$low), max(gdp_scaled$low), length = 496),
  top = seq(min(gdp_scaled$top), max(gdp_scaled$top), length = 496),
  VSD = seq(min(gdp_scaled$VSD), max(gdp_scaled$VSD), length = 496),
  totGVI = seq(min(gdp_scaled$totGVI), max(gdp_scaled$totGVI), length = 496),
  building_height = seq(min(gdp_scaled$building_height), max(gdp_scaled$building_height), length = 496),
  vitality_level = seq(min(gdp_scaled$vitality_level), max(gdp_scaled$vitality_level), length = 496),
  noise_level = seq(min(gdp_scaled$noise_level), max(gdp_scaled$noise_level), length = 496),
  density = seq(min(gdp_scaled$density), max(gdp_scaled$density), length = 496),
  width_index = seq(min(gdp_scaled$width_index), max(gdp_scaled$width_index), length = 496)
)
predicted_values <- predict(optimized_gc_gam, newdata = new_data, type = "response")
shape <- dim(predicted_values)
# Print the shape
print(shape)
new_data <- data.frame(x=as.numeric(x_range),y=as.numeric(predicted_values))
write.table(new_data, file = "D:/WUR/master/MLP/master thesis/data/数据汇总/corr_analysis/find_peaks.txt", sep = "\t", row.names = FALSE)


