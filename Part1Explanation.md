# SeattleMariners
## Part 1 - StrikeCallPrediction

*Objective: Predict the strike call for test.csv*   

**Flow:**    

      - Import  
      - Basic Cleaning (NAs and Redundant Columns)  
      - Feature Engineering   
      - Feature Importance   
      - Final Feature Adjustment/Cleaning  
      - Logistic Regression (Confusion Matrix/Evaluate Accuracy/Predict)   
      - Random Forest (Confusion Matrix/Evaluate Accuracy)   
      - Random Forest Tuning  
      - Random Forest (Predict)    
      - One-hot Encoding  
      - Train and Test Logistic Regression on XGBoost  
      - Evaluate Accuracy  
      - Choose best model, final output  
      
**Explanation:**    

    With a goal to obtain the optimal accuracy on predicting the strike call on the test.csv, it was essential  
    to use comparative assertion on three different models for binary classification - Logistic Regression,  
    Random Forest (Ensembled Decision Trees) and Boosted Logistic Regression using XGBoost. 
  
    For the feature selection, two main methods were used to determine the most effective feature design.  
    One is intuitively being able to tell that the overfitting may be an issue with all the ID Categories  
    except for that of catchers and umpires, as well as the features that aren't necessarily correlated to the classification  
    of strike or ball like spin rate or zone time, etc. The other is the "importance" metric of preliminary random forest  
    method to see the ones that machine likes and doesn't like in predicting the class.   
  
    For feature engineering, binning and grouping were used to release speed and pitch types repectively,  
    as well as the introduction of some new features: StrZoneYD, StrZoneYU, StrZoneXL, StrZoneXR and BorderInclusion.  
    After testing the new features through the preliminary random forest, the importance score appeared as about  
    ~8000 for BorderInclusion, almost doubling that of plate_height and plate_side. BorderInclusion is a binary  
    variable that returns "yes" for the pitch that comes inside the designated strike zone (constructed by taking  
    the minimum and maximum height and width of the pitch locations by Umpire ID and Batter-Handedness), and no for  
    otherwise case. 
  
    Logistic Regression was trained on the dataset with Release Height, Release Side, Vertical Break, Horizontal Break,  
    Plate Height, Plate Side, Ball Count, StrZone(YD,YU,XL,XR) and Border Inclusion. The test prediction was made,  
    and according to the confusion matrix, the accuracy was 0.8293.

    Random Forest was trained on the 50000 sampled dataset with Release Speed, Release Height, Release Side, Vertical Break,  
    Horizontal Break, Plate Height, Plate Side, Ball Count, StrZone(YD,YU,XL,XR) and Border Inclusion. The test prediction   
    was made, and according to the confusion matrix, the accuracy was 0.921926.

    Boosted Logistic Regression was trained on the dataset with  Catcher ID, Release Speed, Release Height, Release Side,   
    Vertical Break, Horizontal Break, Plate Height, Plate Side, Ball Count, StrZone(YD,YU,XL,XR) and Border Inclusion.  
    The train error was 0.058164. Since the XGBoost logistic regression had the lowest error, we choose the last model to  
    be the predictor. 

    Any data with the NA in the predictor variables were handled separately according to the available set of predictors through  
    Random Forest, then added onto the output.
  
  **Further Implication:**   
  
    For even more precise is_strike result, more balanced number of samples for catcher_id is needed, as I couldn't group the  
    number down to less than 53 levels (is what you need in order to have it in RF) based on the strike zone similarities.   
    This is important since I believe catcher's pitch framing is what can be of great impact on the pitch call, along with  
    the umpire's characteristic.
    With Batter Height info, more accurate strike zone can be constructed since the vertical range of a given batter's 
    strike zone is very much dependent on his height. With enough at-bats per batter, we can construct individually tailored  
    strike zone and tune more accurate BorderInclusion variable. 
  
  
      
      
