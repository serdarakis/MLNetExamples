namespace MLNetExamples.Console;

using Microsoft.ML;

public static class HousePricePredictionWithTrainedModel
{
    public static void TrainModel()
    {
        var mlContext = new MLContext();
        
        HouseData[] houseData =
        [
            new() { Size = 100F, Price = 1.2F },
            new() { Size = 115.5F, Price = 1.9F },
            new() { Size = 86F, Price = 1.0F },
            new() { Size = 98.8F, Price = 1.2F }
        ];
        IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

        // 2. Specify data preparation and model training pipeline
        var pipeline = mlContext.Transforms.Concatenate(
                "Features", "Size")
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Price", maximumNumberOfIterations: 100, featureColumnName: "Features"));

        // 3. Train model
        var model = pipeline.Fit(trainingData);
        
        // 4. Save Model
        mlContext.Model.Save(model, trainingData.Schema, "HousePriceModel.zip");
    }
    
    public static float Predict(float size)
    {
        // 1. Load model
        var mlContext = new MLContext();
        var model = mlContext.Model.Load("HousePriceModel.zip", out _);
        
        // 2. Make a prediction
        var houseSize = new HouseData { Size = size };
        var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(houseSize);

        return price.Price;
    }
}