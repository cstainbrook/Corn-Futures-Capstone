# Corn-Futres-Capstone
My capstone project at Galvanize, I aimed to predict a range of potential corn prices using data on the corn market and the weather in key producing states. I was able to achieve sufficient results for implementing a hedging strategy, although overall returns would not be high enough to warrant speculation.  For current predictions and a prettier presentation, please visit my website: cooperscorn.com

# Summary
Going into the project, I had two main goals:
* Predict the direction of the price movement to lend farmers confidence for hedging their crop production
* Develop a model that would earn acceptable investment returns for traders

To achieve these objectives, I developed several neural networks, which relied on market data as inputs.  While I largely accomplished my goal of predicting direction, investment returns would be substandard for all but those with a very low cost of capital.

# Results
I evaluated my models using RMSE, hit rate (the percentage of times I predicted the correct direction of price movement), and trading return.  The graph below shows my 6-month prediction model for both in-sample and out-of-sample results.  The shaded regions show a confidence interval, as represented by 2 x RMSE.

![Image of Results]
(Figures/neuralnet_six_month.png)

I am most pleased with my hit rate results, correctly identifying the direction of movement 75% of the time when the current price is at least 10% different from my predicted price.  This climbs to 98% when the current price is at least 30% different from my predicted price.


Through the EDA process, I also uncovered some interesting phenomena in the data.  As shown below, the introduction of corn syrup and E85 produced demand shocks that appear to be correlated with a steep increase in price.  One can immediately see the dynamics of supply and demand over time in the second graph, as the increase in production has led to a consistent decline in inflation adjusted price.

![Image of Demand Shocks]
(Figures/inflation_adjusted_demand_shocks.png)


# Technology Used
* Python Packages
    * Keras
    * Pandas
    * Numpy
    * SciPy
    * StatsModels
    * IbPy
    * Flask

* AWS EC2

# Data
While not all datasets collected were used as features for the final models, the following was collected.

### Interactive Brokers (Daily Data)
* Corn Futures Prices
* Soybean Futures Prices
* Oil Futures Prices
* $USD Exchange Rate Index

### USDA (Yearly and Quarterly Data)
* Corn Supply Levels
* Acreage Planted
* Yield Estimates

### NOAA (Daily Data)
* Precipitation and Temperatures for Iowa, Illinois, Nebraska, Minnesota, and Indiana
* ONI Index

# Model
The final models I developed are feed-forward neural networks, which use either 6-month or 3-month lagged data as features.  The structure of the nets is quite simple, consisting of 3 layers, the hidden layer having a linear activation function and the output layer having a sigmoid activation function.  I used the lagged features rather than current data to simulate forecasting 3 or 6 months out with the information that is readily available.

# Website
After finishing the modeling phase of the project, I developed a website, cooperscorn.com, to display the results of my research as well as current predictions.  The website was built using flask bootstrapping and is hosted on an Amazon EC2 instance.

# Future Steps
* Monitor my predictions to ensure outcomes are comparable with current out-of-sample forecasts
* Speak with futures traders to identify other potentially import features
