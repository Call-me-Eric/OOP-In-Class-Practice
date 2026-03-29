#include<string>
#include<cctype>
#include<sstream>
#include<iostream>
using namespace std;

int classExample(string str){
    int sum=0;
    int i=0;
    while(i<str.size()){
        if(isdigit(str[i])){
            int num=0;
            while(i<str.size()&&isdigit(str[i])){
                num=num*10+(str[i]-'0');
                i++;
            }
            sum+=num;
        }
        else{
            i++;
        }
    }
    return sum;
}

void extractedQuote(){
    string first_name,last_name;
    cin>>first_name>>last_name;
    string quote,word;
    while(cin>>word){
        quote=quote+word+" ";
        if(cin.peek()=='/n'){
            break;
        }
    }
    cout<<first_name<<" "<<last_name<<" said this: "<<quote<<endl;

    //string sentence;
    //getline(cin,sentence);
    //stringstream ss(sentence);
    //string first_name,last_name,quote;
    //ss>>first_name>>last_name;
    //getline(ss,quote);
    //cout<<first_name<<" "<<last_name<<" said this:"<<quote<<endl;

}
