#include <stdio.h>  
  
int main()  
{  
    int i = 100; // 从333开始循环  
    for (; i < 334; i++) // 假设循环到399  
    {  
        int num = i; // 使用循环变量作为num的值  
        int number = 2 * num;  
        int count = 3 * num;  
  
        // 分解num, number, count  
        int hundred = num / 100;  
        int hundred2 = number / 100;  
        int hundred3 = count / 100;  
        int tenth = (num / 10) % 10; // 更正计算十位的方法  
        int tenth2 = (number / 10) % 10;  
        int tenth3 = (count / 10) % 10;  
        int per = num % 10; // 更正计算个位的方法  
        int per2 = number % 10;  
        int per3 = count % 10;  
  
        // 检查所有值是否非零且互不相等  
        int allDifferent = 1; // 假设所有值都不同  
        if (hundred == hundred2 || hundred == hundred3 || hundred == tenth || hundred == tenth2 || hundred == tenth3 || hundred == per || hundred == per2 || hundred == per3 || hundred2 == hundred3 || hundred2 == tenth || hundred2 == tenth2 || hundred2 == tenth3 || hundred2 == per || hundred2 == per2 || hundred2 == per3 || hundred3 == tenth || hundred3 == tenth2 || hundred3 == tenth3 || hundred3 == per || hundred3 == per2 || hundred3 == per3 ||tenth == tenth2 || tenth == tenth3 || tenth == per || tenth == per2 || tenth == per3 || tenth2 == tenth3 || tenth2 == per || tenth2 == per2 || tenth2 == per3 || tenth3 == per || tenth3 == per2 || tenth3 == per3 || per == per2 || per == per3 || per2 ==per3) 
		{  
            allDifferent = 0; // 至少有一个值相同  
        }  
  
        if (allDifferent!=0) {  
            printf("%d %d %d\n", num, number, count);  
        }  
    }  
    return 0;  
}
