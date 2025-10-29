namespace Spectra.ImageProcessing;

public class ImageProcessingInstance
{
    public static void RunImageProcessing()
    {
        var prjPath = "D:\\DotNetProjects10\\Spectra\\Spectra.ImageProcessing\\Models\\";

        var session = new InferenceSession(prjPath + "mobilenetv2-10.onnx");
        string imagePath = prjPath + "ButterflyFish.jpg";
        var inputTensor = PreprocessImage(imagePath);

        var inputName = session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        int topIndex = Array.IndexOf(output, output.Max());

        string[] labels = File.ReadAllLines(prjPath + "synset.txt");
        string predictedLabel = labels.Length > topIndex ? labels[topIndex] : "Unknown";

        Console.WriteLine($"Predicted class: {predictedLabel}");
    }

    static DenseTensor<float> PreprocessImage(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        image.Mutate(x => x.Resize(224, 224));

        var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

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
