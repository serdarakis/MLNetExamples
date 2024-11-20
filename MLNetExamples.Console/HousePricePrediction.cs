namespace MLNetExamples.Console;

using Microsoft.ML;

public static class HousePricePrediction
{
    public static float Predict(float size)
    {
        var mlContext = new MLContext();
        // 1. Import or create training data
        HouseData[] houseData =
        [
            new() { Size = 100F, Price = 1.2F },
            new() { Size = 115.5F, Price = 1.9F },
            new() { Size = 86F, Price = 1.0F },
            new() { Size = 98.8F, Price = 1.2F }
        ];
        var trainingData = mlContext.Data.LoadFromEnumerable(houseData);

        // 2. Specify data preparation and model training pipeline with Stochastic Dual Coordinate Ascent (SDCA) 
        var pipeline = mlContext.Transforms.Concatenate(
                "Features", "Size")
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Price", maximumNumberOfIterations: 100, featureColumnName: "Features"));

        // 3. Train model
        var transformer = pipeline.Fit(trainingData);

        // 4. Make a prediction
        var houseSize = new HouseData { Size = size };
        var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>
            (transformer).Predict(houseSize);

        return price.Price;
    }
}