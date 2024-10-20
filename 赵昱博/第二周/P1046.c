#include <stdio.h>

int main()
{
    int a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=0;
    scanf("%d" "%d" "%d" "%d" "%d" "%d" "%d" "%d" "%d" "%d",&a1,&a2,&a3,&a4,&a5,&a6,&a7,&a8,&a9,&a10);
    int help=30;
    int height=0;
    scanf("%d",&height);
    int total=height+help;
    int tnumber=0;
    if(total>=a1)
    {
        tnumber++;
    }

    if(total>=a2)
    {
        tnumber++;
    }

    if(total>=a3)
    {
        tnumber++;
    }

    if(total>=a4)
    {
        tnumber++;
    }

    if(total>=a5)
    {
        tnumber++;
    }

    if(total>=a6)
    {
        tnumber++;
    }

    if(total>=a7)
    {
        tnumber++;
    }

    if(total>=a8)
    {
        tnumber++;
    }

    if(total>=a9)
    {
        tnumber++;
    }

    if(total>=a10)
    {
        tnumber++;
    }
    
    printf("%d",tnumber);
    return 0;
}
