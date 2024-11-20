using Microsoft.ML.Data;

namespace MLNetExamples.Console;

using System;
using Microsoft.ML;

public static class HousePricePredictionWithONNXModel
{
    public static void TrainModel()
    {
        var mlContext = new MLContext();

        HouseData[] houseData =
        [
            new() { Size = 1.1F, Price = 1.2F },
            new() { Size = 1.9F, Price = 2.3F },
            new() { Size = 2.8F, Price = 3.0F },
            new() { Size = 3.4F, Price = 3.7F }
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

        File.Delete("HousePriceModel.onnx");
        using var onnx = File.Open("HousePriceModel.onnx", FileMode.OpenOrCreate, FileAccess.Write);
        mlContext.Model.ConvertToOnnx(model, trainingData, onnx);
    }

    public static void Predict(float size)
    {
        // 1. Load model
        var mlContext = new MLContext();
        var pipeline = mlContext.Transforms.ApplyOnnxModel(
            modelFile: "HousePriceModel.onnx",
            inputColumnNames: new[] { "Size" }, // Match input column
            outputColumnNames: new[] { "Score.output"}); // 

        
        // 2. Make a prediction
        var houseSize = new List<HouseData> { new() { Size = size } };

        var testDataView = mlContext.Data.LoadFromEnumerable(houseSize);
        
        // 5. Transform test data using the ONNX model
        var transform = pipeline.Fit(testDataView);
        var transformedData = transform.Transform(testDataView);
        // mlContext.Model.Save(transform, testDataView.Schema, "HousePriceModel2.zip");
        // [Size.output,Price.output,Features.output,Score.output]
        // 6. Extract predictions
        //var predictions = mlContext.Data.CreateEnumerable<Prediction2>(transformedData, reuseRowObject: false);
        // var result = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(transformedData).Predict(houseSize);
        var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, Prediction2>(transform);
        var houseData = new HouseData { Size = size};
        var result = onnxPredictionEngine.Predict(houseData);
        
        
        // Console.WriteLine($"Predicted price for size: {houseSize.Size} = {price.Price * 100:C}k");
    }
    
    public static void Predict2(float size)
    {
        // 1. Load model
        var mlContext = new MLContext();
        mlContext.Model.Load("HousePriceModel.onnx", out var model);
        var onnxEstimator = mlContext.Transforms.ApplyOnnxModel("HousePriceModel.onnx");
        // var pipeline = mlContext.Transforms.ApplyOnnxModel(
        //     modelFile: "HousePriceModel.onnx",
        //     inputColumnNames: new[] { "Size" }, // Match input column
        //     outputColumnNames: new[] { "Score.output"}); // 

        
        // 2. Make a prediction
        var houseSize = new List<HouseData> { new() { Size = size } };

        var testDataView = mlContext.Data.LoadFromEnumerable(houseSize);
        
        // 5. Transform test data using the ONNX model
        var onnxTransformer = onnxEstimator.Fit(testDataView);
        var output = onnxTransformer.Transform(testDataView);
        var onnxOutScores = mlContext.Data.CreateEnumerable<Prediction2>(output, reuseRowObject: false);
        // var onnxOutput = onnxTransformer.Transform(testDataView);
        // mlContext.Model.Save(transform, testDataView.Schema, "HousePriceModel2.zip");
        // [Size.output,Price.output,Features.output,Score.output]
        // 6. Extract predictions
        //var predictions = mlContext.Data.CreateEnumerable<Prediction2>(transformedData, reuseRowObject: false);
        // var result = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(transformedData).Predict(houseSize);
        // var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, Prediction2>(transform);
        // var houseData = new HouseData { Size = size};
        // var result = onnxPredictionEngine.Predict(houseData);
        //
        //
        // Console.WriteLine($"Predicted price for size: {houseSize.Size} = {price.Price * 100:C}k");
    }
}

public class Prediction2
{
    // public float Price { get; set; }
    // [ColumnName("Price.output")] public float PriceO { get; set; }
    [ColumnName("Score.output")] public float ScoreO { get; set; }
    [ColumnName("Features.output")] public float Features { get; set; }
    // [ColumnName("Score")] public float Score { get; set; }
    // [ColumnName("Price")] public float Price { get; set; }
    [ColumnName("Price.output")] public float PriceO { get; set; }
}