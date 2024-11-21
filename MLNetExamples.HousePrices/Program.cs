
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace MLNetExamples.HousePrices.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            MLNetExamples_HousePrices.ModelInput sampleData = new MLNetExamples_HousePrices.ModelInput()
            {
                Size = 1200F,
                Bedrooms = 3F,
                Bathrooms = 2F,
                YearBuilt = 2005F,
            };


            Console.WriteLine("Using model to make single prediction -- Comparing actual Price with predicted Price from sample data...\n\n");


            Console.WriteLine($"Size: {1200F}");
            Console.WriteLine($"Bedrooms: {3F}");
            Console.WriteLine($"Bathrooms: {2F}");
            Console.WriteLine($"YearBuilt: {2005F}");
            Console.WriteLine($"Price: {250000F}");


            // Make a single prediction on the sample data and print results
            var predictionResult = MLNetExamples_HousePrices.Predict(sampleData);

            Console.WriteLine($"\n\nPredicted Price: {predictionResult.Score}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}