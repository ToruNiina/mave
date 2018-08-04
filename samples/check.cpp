#include <mave/mave.hpp>
#include <iostream>

int main()
{
    std::cout << mave::supported_instructions() << std::endl;
    return 0;
}
