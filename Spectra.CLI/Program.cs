class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Args empty");
        }

        ImageProcessingInstance.RunImageProcessing();
    }
}