namespace Fixture
{
    public class App
    {
        static int Process(int x)
        {
            return Util.Helper(x);
        }

        static int Run()
        {
            return new Engine().Process(Process(1));
        }
    }
}
