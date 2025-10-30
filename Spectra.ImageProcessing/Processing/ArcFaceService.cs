namespace Spectra.ImageProcessing.Processing;

public class ArcFaceService : IDisposable
{
    private readonly InferenceSession _session;
    public string InputName { get; }
    public string OutputName { get; }

    public ArcFaceService(SessionOptions options = null)
    {
        var path = "D:\\DotNetProjects10\\Spectra\\Spectra.ImageProcessing\\Models\\ArcFace\\";
        var modelFullPath = path + "arcfaceresnet100-11-int8.onnx";
        _session = options == null ? new InferenceSession(modelFullPath) : new InferenceSession(modelFullPath, options);

        // Query metadata (helps avoid hardcoding input/output names)
        var inputs = _session.InputMetadata;
        if (inputs.Count == 0) throw new Exception("Model has no inputs?");
        InputName = inputs.Keys.First();

        var outputs = _session.OutputMetadata;
        OutputName = outputs.Keys.First();
    }

    // main method: feed a cropped face image (stream) and get L2-normalized embedding
    // preprocessing toggles:
    //   targetSize: typically 112
    //   swapToBgr: whether to change channel order to BGR (set true if model expects BGR)
    //   normalizeBy255: if true uses pixel/255.0; otherwise uses (pixel - 127.5)/128
    public async Task<float[]> GetEmbedding(Stream imageStream, int targetSize = 112, bool swapToBgr = false, bool normalizeBy255 = false)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(imageStream);

        // Resize to square target
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(targetSize, targetSize),
            Mode = ResizeMode.Stretch
        }));

        // Prepare tensor: shape [1, 3, H, W] (float32)
        var tensor = new DenseTensor<float>([1, 3, targetSize, targetSize]);

        // Fill tensor
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < targetSize; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < targetSize; x++)
                {
                    var px = pixelRow[x];
                    // px.R/G/B are bytes 0..255
                    float r = px.R;
                    float g = px.G;
                    float b = px.B;

                    // channel order: model may expect RGB or BGR
                    float c0 = swapToBgr ? b : r;
                    float c1 = g;
                    float c2 = swapToBgr ? r : b;

                    // normalization
                    if (normalizeBy255)
                    {
                        c0 /= 255f; c1 /= 255f; c2 /= 255f;
                    }
                    else
                    {
                        // common ArcFace normalization: (x - 127.5) / 128
                        c0 = (c0 - 127.5f) / 128f;
                        c1 = (c1 - 127.5f) / 128f;
                        c2 = (c2 - 127.5f) / 128f;
                    }

                    tensor[0, 0, y, x] = c0;
                    tensor[0, 1, y, x] = c1;
                    tensor[0, 2, y, x] = c2;
                }
            }
        });

        // Run inference
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var outTensor = results.First(r => r.Name == OutputName).AsTensor<float>();
        float[] embedding = [.. outTensor];

        // L2-normalize
        float norm = (float)Math.Sqrt(embedding.Select(v => v * v).Sum());
        if (norm > 0)
        {
            for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;
        }

        return embedding;
    }

    public static async Task<float> CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Lengths differ");
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / ((float)Math.Sqrt(na) * (float)Math.Sqrt(nb));
    }

    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}
