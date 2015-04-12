## Monte Carlo Option Pricing

# compile code
mvn package

# execute project
spark-submit --class com.lampalork.mcoptionpricing.MonteCarloOptionPricing --master local target\montecarlo-optionpricing-0.0.1-SNAPSHOT.jar 500000 5 
