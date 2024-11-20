namespace MLNetExamples.Console;

using System;
using Microsoft.ML;

public static class HousePricePrediction
{
    public static void Predict(float size)
    {
        MLContext mlContext = new MLContext();

        // 1. Import or create training data
        HouseData[] houseData =
        {
            new() { Size = 1.1F, Price = 1.2F },
            new() { Size = 1.9F, Price = 2.3F },
            new() { Size = 2.8F, Price = 3.0F },
            new() { Size = 3.4F, Price = 3.7F }
        };
        IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

        // 2. Specify data preparation and model training pipeline
        var pipeline = mlContext.Transforms.Concatenate(
                "Features", "Size")
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Price", maximumNumberOfIterations: 100, featureColumnName: "Features"));

        // 3. Train model
        var model = pipeline.Fit(trainingData);


        // 4. Make a prediction
        var houseSize = new HouseData { Size = size };
        var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(houseSize);

        Console.WriteLine($"Predicted price for size: {houseSize.Size} = {price.Price * 100:C}k");

        // Predicted price for size: 2500 sq ft= $261.98k
    }
}