
library(gridExtra)
library(tidyverse)

traindf <- read.csv("2020-train.csv")

catcherdf <- traindf %>%
  filter(catcher_id == "f06c9fdf")

# Framing Ability - Compare with BoundaryInclusion and see how many strikes/balls were given
Str <- catcherdf %>%
  filter(pitch_call == "StrikeCalled")

temp <- Str %>%
     group_by(umpire_id,batter_side) %>%
     summarize(L = mean(StrZoneXL), R = mean(StrZoneXR), D = mean(StrZoneYD), U = mean(StrZoneYU))
tempR <- temp %>%
  filter(batter_side == "Right")
tempL <-temp %>%
  filter(batter_side == "Left")

mean(tempR$L)
mean(tempR$R)
mean(tempR$U)
mean(tempR$D)

MeanR <- c(-1.215534,1.086866,3.654807,1.238012)

mean(tempL$L)
mean(tempL$R)
mean(tempL$U)
mean(tempL$D)

MeanL <- c(-1.017976,1.181128,3.679435,1.298374)

pitchplotR <- catcherdf %>%
  filter(batter_side == "Right") %>%
  ggplot(aes(x = plate_side, y = plate_height, colour = pitch_call)) +
  geom_point(alpha = 0.3)  +
  geom_rect(mapping = aes(xmin = -1.22,xmax = 1.09, ymin = 1.24, ymax =3.65),size =1, fill = NA,colour = "black", alpha = 0.1) +
  labs(x = "Horizontal Placement", y = "vertical Placement", title = "Received Pitches for Right-handed Batters", labels = c("Ball","Strike"),color = "Pitch Call" )
  

pitchplotL <- catcherdf %>% 
  filter(batter_side == "Left") %>%
  ggplot(aes(x = plate_side, y = plate_height, colour = pitch_call)) +
  geom_point(alpha = 0.3) +
  geom_rect(mapping = aes(xmin = -1.02,xmax = 1.18, ymin = 1.3, ymax =3.68),size =1, fill = NA,colour = "black", alpha = 0.1) +
  labs(x = "Horizontal Placement", y = "vertical Placement", title = "Received Pitches for Left-handed Batters", labels = c("Ball","Strike"),color = "Pitch Call" )

# Pitch Types and Pitch Strike Breakdown
PitchBreakdown <- catcherdf %>%
  ggplot(aes(x = BallCount, fill = pitch_type)) +
  geom_bar() +
  labs(X = "Ball Count", y = "Number", colour = "Pitch Type",title = "Received Pitch Types by Ball Count")

PitchBallStrike <- catcherdf %>%
  ggplot(aes(x = BallCount, fill = pitch_call)) +
  geom_bar() +
  labs(X = "Ball Count", y = "Number", title = "Received Pitch Calls by Ball Count",color = "Pitch Call" )






