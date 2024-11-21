
Prerequisites

Install the .NET SDK.

Install the ML.NET CLI tool

    dotnet tool install -g mlnet

Add the ML.NET CLI to your PATH environment variable:

    export PATH="$PATH:<Your path>"

__Example 1: Sentiment Analysis__
Dataset

The sentiment analysis model uses the yelp_labelled.txt dataset, which contains two columns:

    Text: Customer review text.
    Label: Sentiment (0 = Negative, 1 = Positive).

__Train the Model__

Run the following command to train a classification model:

mlnet classification --dataset "yelp_labelled.txt" --label-col 1 --has-header false --train-time 10 --name "MLNetExamples.SentimentModel"

__Output__

    A trained sentiment analysis model.
    A generated C# project for consuming the model.

__Example 2: House Price Prediction__

Dataset

The house price prediction model uses the house_prices.csv dataset with the following columns:

    Size: Square footage of the house.
    Bedrooms: Number of bedrooms.
    Bathrooms: Number of bathrooms.
    YearBuilt: Year the house was built.
    Price: The target column, representing the selling price of the house.

__Train the Model__

Run the following command to train a regression model:

    mlnet regression --dataset "house_prices.csv" --label-col 4 --has-header true --train-time 10 --name "MLNetExamples.HousePrices"

__Output__

    A trained house price prediction model.
    A generated C# project for consuming the model.

__Notes__

    Both examples generate a ready-to-use C# console application with the trained model and helper code for predictions.
    You can modify the datasets or parameters (like training time) to explore different results.

__Getting Started__

    Clone this repository.
    Train the models using the provided commands.
    Explore the generated code to understand how ML.NET integrates into .NET projects.

__Feedback & Contribution__

Feel free to open issues or contribute improvements to this repository. Happy coding!