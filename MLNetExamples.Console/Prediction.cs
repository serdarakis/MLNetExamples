using Microsoft.ML.Data;

namespace MLNetExamples.Console;

public class Prediction
{
    [ColumnName("Score")] public float Price { get; set; }
}