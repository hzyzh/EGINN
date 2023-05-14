public class A{
    public void caller(){
        int ans1 = target(10);
        int ans2 = target(20);
        System.out.print(ans1 + ans2);
    }

    public int target(int a){
        int result = 0;
        int i = 1;
        while(i < a){
            result = result + callee1(i);
            result = result + callee2(i);
            i = i - 1;  //bug position
        }
        return result;
    }

    public int callee1(int a){
        if(a % 2 == 0){
            return a / 2;
        }
        else{
            return a - 7;
        }
    }

    public int callee2(int a){
        return (a + 5) * (a / 3 + 1);
    }
}