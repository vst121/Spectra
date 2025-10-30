using SixLabors.ImageSharp.Advanced;

namespace Spectra.ImageProcessing.Processing;

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


    public (float age, string gender, float genderConfidence) PredictAgeGender(Stream selectedImage)
    {
        // Preprocess image -> tensor
        var tensor = ImageToTensor(selectedImage, _inputWidth, _inputHeight);

        float age, genderConf;
        string gender;

        age = GetAge(tensor);
        (gender, genderConf) = GetGender(tensor);

        return (age, gender, genderConf);
    }

    public float GetAge(DenseTensor<float> tensor)
    {
        var inputMeta = _sessionForAge.InputMetadata;
        var firstInputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(firstInputName, tensor)
        };

        using var results = _sessionForAge.Run(inputs);

        var outList = results.ToList();

        float age = -1f;

        if (outList.Count == 1 && outList[0].Value is DenseTensor<float> t)
        {
            var values = t.ToArray();

            if (values.Length == 8)
            {
                var ageProbs = Softmax(values);
                var maxIndex = Array.IndexOf(ageProbs, ageProbs.Max());
                int[] ageMidpoints = [1, 5, 10, 18, 28, 40, 50, 65];
                age = ageMidpoints[maxIndex];
            }
        }

        return age;
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

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        var sum = exps.Sum();
        return [.. exps.Select(x => x / sum)];
    }

    private static DenseTensor<float> ImageToTensor(Stream selectedImage, int width, int height)
    {
        using var img = Image.Load<Rgb24>(selectedImage);
        img.Mutate(x => x.Resize(new ResizeOptions { Size = new SixLabors.ImageSharp.Size(width, height), Mode = ResizeMode.Crop }));


        // Model normalization: many ONNX vision models expect (pixel - mean)/scale with mean around 127 and scale 255
        // Adjust if the model README specifies other values.
        float mean = 127f;
        float scale = 255f;


        var tensor = new DenseTensor<float>([1, 3, height, width]);


        for (int y = 0; y < height; y++)
        {
            var span = img.Frames.RootFrame.DangerousGetPixelRowMemory(y).Span;
            for (int x = 0; x < width; x++)
            {
                var p = span[x];
                // channel order: R,G,B -> put into tensor as [0,R],[1,G],[2,B]
                tensor[0, 0, y, x] = (p.R - mean) / scale;
                tensor[0, 1, y, x] = (p.G - mean) / scale;
                tensor[0, 2, y, x] = (p.B - mean) / scale;
            }
        }


        return tensor;
    }

    public void Dispose()
    {
        _sessionForAge.Dispose();
        _sessionForGender.Dispose();
        GC.SuppressFinalize(this);
    }
}