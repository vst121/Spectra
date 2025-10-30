namespace Spectra.ImageProcessing.Processing;

public class ClassificationService
{
    public static Task<string> DoClassification(Stream selectedImage)
    {
        var modelPath = "D:\\DotNetProjects10\\Spectra\\Spectra.ImageProcessing\\Models\\Classification\\";

        var session = new InferenceSession(modelPath + "mobilenetv2-10.onnx");
        var inputTensor = PreprocessImage(selectedImage);

        var inputName = session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results[0].AsEnumerable<float>().ToArray();

        int topIndex = Array.IndexOf(output, output.Max());

        string[] labels = File.ReadAllLines(modelPath + "synset.txt");
        string predictedLabel = labels.Length > topIndex ? labels[topIndex] : "Unknown";

        return Task.FromResult(predictedLabel);
    }
    static DenseTensor<float> PreprocessImage(Stream selectedImage)
    {
        using var image = Image.Load<Rgb24>(selectedImage);
        image.Mutate(x => x.Resize(224, 224));

        var input = new DenseTensor<float>([1, 3, 224, 224]);

        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                var pixel = image[x, y];
                input[0, 0, y, x] = pixel.R / 255f;
                input[0, 1, y, x] = pixel.G / 255f;
                input[0, 2, y, x] = pixel.B / 255f;
            }
        }

        return input;
    }

}
