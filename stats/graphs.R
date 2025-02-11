library(ggplot2)
library(scales)

# Read the filtered vector count
data <- read.table("../materials/vectors.txt", skip = 1, stringsAsFactors = FALSE)

# Extract the first token and the first number
df <- data.frame(
  rank = 1:nrow(data),
  token = data$V1,
  frequency = as.numeric(data$V2),
  percentile = as.numeric(data$V3)
)
df$theoretical_zipf <- df$rank^(-1) * max(df$frequency)

# Graphs to plot the ranks
ggplot(df, aes(x = rank, y = frequency)) +
  geom_point(aes(color = "real")) +
  geom_line(aes(color = "real")) +
  geom_line(aes(y = theoretical_zipf, color = "estimated"), linetype = "dashed") +
  labs(title = "Sampled frequencies from Corpus Spoken Dutch",
       x = "Rank",
       y = "Frequency",
       color = "Legend") +
  scale_color_manual(values = c("real" = "black", "estimated" = "red"), labels=
                       c("Estimated value", "Real value"))

ggplot(df, aes(x = rank, y = percentile)) +
  geom_point(aes()) +
  geom_line(aes()) +
  labs(title = "Sampled frequencies from Corpus Spoken Dutch",
       x = "Rank",
       y = "Percentile") +
  scale_y_continuous(labels = scales::percent)
