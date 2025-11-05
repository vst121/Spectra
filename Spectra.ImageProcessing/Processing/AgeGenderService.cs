namespace Spectra.ImageProcessing.Processing;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class AgeGenderService : IDisposable
{
    private readonly InferenceSession _sessionForAge, _sessionForGender;
    private readonly int _inputWidth = 224; 
    private readonly int _inputHeight = 224;

    public AgeGenderService()
    {
        var path = "D:\\DotNetProjects10\\Spectra\\Spectra.ImageProcessing\\Models\\AgeGender\\";
        var modelForAgeFullPath = path + "age_googlenet.onnx";
        var modelForGenderFullPath = path + "gender_googlenet.onnx";

        _sessionForAge = new InferenceSession(modelForAgeFullPath);
        _sessionForGender = new InferenceSession(modelForGenderFullPath);
    }


    public (string age, string gender, float genderConfidence) PredictAgeGender(Stream selectedImage)
    {
        // Preprocess image -> tensor
        var tensor = ImageToTensor(selectedImage, _inputWidth, _inputHeight);

        var age = GetAge(tensor);
        (var gender, float genderConfidence) = GetGender(tensor);

        return (age, gender, genderConfidence);
    }

    public (string gender, float genderConfidence) GetGender(DenseTensor<float> tensor)
    {
        var inputMeta = _sessionForGender.InputMetadata;
        var firstInputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(firstInputName, tensor)
        };

        using var results = _sessionForGender.Run(inputs);

        var outList = results.ToList();

        float genderConf = 0f;
        string gender = "unknown";

        if (outList.Count == 1 && outList[0].Value is DenseTensor<float> t)
        {
            var values = t.ToArray();

            if (values.Length == 2)
            {
                var genderProbs = Softmax(values);
                genderConf = genderProbs[1];
                gender = genderProbs[1] > 0.5f ? "female" : "male";
            }
        }

        return (gender, genderConf);
    }

    public string GetAge(DenseTensor<float> tensor)
    {
        var inputMeta = _sessionForAge.InputMetadata;
        var firstInputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(firstInputName, tensor)
        };

        using var results = _sessionForAge.Run(inputs);

        var outList = results.ToList();

        string age = "Unknown!";

        if (outList.Count == 1 && outList[0].Value is DenseTensor<float> t)
        {
            var values = t.ToArray();

            if (values.Length == 8)
            {
                var ageProbs = Softmax(values);

                float[] ageCenters = [1, 5, 10, 18, 28, 40, 50, 80];
                string[] ageRanges = ["0 - 2", "4 - 6", "8 - 13", "15 - 20", "25 - 32", "38 - 43", "48 - 53", "60 - 100"];

                float predictedAge = 0f;
                for (int i = 0; i < 8; i++)
                    predictedAge += ageProbs[i] * ageCenters[i];

                var maxIndex = Array.IndexOf(ageProbs, ageProbs.Max());
                string predictedRange = ageRanges[maxIndex];

                age = $"{predictedRange} (≈ {predictedAge:F1} years)";
            }
        }

        return age;
    }


    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        var sum = exps.Sum();
        return [.. exps.Select(x => x / sum)];
    }

    private static DenseTensor<float> ImageToTensor(
        Stream imageStream, int width, int height,
        bool useImageNetNormalization = false)
    {
        using var image = Image.Load<Rgb24>(imageStream);
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new SixLabors.ImageSharp.Size(width, height),
            Mode = ResizeMode.Crop
        }));

        var tensor = new DenseTensor<float>([1, 3, height, width]);

        float[] mean = useImageNetNormalization ? [0.485f, 0.456f, 0.406f] : [0f, 0f, 0f];
        float[] std = useImageNetNormalization ? [0.229f, 0.224f, 0.225f] : [1f, 1f, 1f];

        for (int y = 0; y < height; y++)
        {
            var pixelRow = image.DangerousGetPixelRowMemory(y).Span;
            for (int x = 0; x < width; x++)
            {
                ref var p = ref pixelRow[x];
                tensor[0, 0, y, x] = ((p.B / 255f) - mean[0]) / std[0];
                tensor[0, 1, y, x] = ((p.G / 255f) - mean[1]) / std[1];
                tensor[0, 2, y, x] = ((p.R / 255f) - mean[2]) / std[2];
            }
        }

        return tensor;
    }

    public void Dispose()
    {
        _sessionForGender.Dispose();
        GC.SuppressFinalize(this);
    }
}