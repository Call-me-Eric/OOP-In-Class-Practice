#include<iostream>
#include<unordered_map>
#include<unordered_set>
#include<string>
#include<vector>
#include<algorithm>
using namespace std;
class Student {
    private:
        string name;
        unordered_set<string> friends;
    public:
        Student() {}
        Student(string n) : name(n) {}
        void addFriend(const string& f)
        {
            friends.insert(f);
        }
        void deleteFriend(const string& f)
        {
            if(friends.find(f) != friends.end())friends.erase(f);
        }
        int getFriendCount() const { return friends.size(); }
        unordered_set<string> getFriendList() const { return friends; }
};
class FriendSystem {
    private:
        unordered_map<string, Student> students;
    public:
        void addRelation(const string& A, const string& B)
        {
            students[A].addFriend(B);
            students[B].addFriend(A);
        }
        void deleteRelation(const string& A, const string& B)
        {
            students[A].deleteFriend(B);
            students[B].deleteFriend(A);
        }
        void queryFriendCount(const string& A)
        {
            cout << students[A].getFriendCount() << endl;
        }
        void queryCommonFriends(const string& A, const string& B)
        {
            unordered_set<string> friendsA = students[A].getFriendList();
            unordered_set<string> friendsB = students[B].getFriendList();
            int common = 0;
            for (const string& f : friendsA)
            {
                if (friendsB.find(f) != friendsB.end())
                {
                    common++;
                    cout << f << " ";
                }
            }
            if(common == 0)cout << "none";
            cout << endl;
        }
};
int main() {
    int n;
    cin >> n;
    FriendSystem sys;
    for (int i = 0; i < n; ++i) {
        string op;
        cin >> op;
        if (op == "add") {
            string A, B;
            cin >> A >> B;
            sys.addRelation(A, B);
        } else if (op == "delete") {
            string A, B;
            cin >> A >> B;
            sys.deleteRelation(A, B);
        } else if (op == "query") {
            string A;
            cin >> A;
            sys.queryFriendCount(A);
        } else if (op == "common") {
            string A, B;
            cin >> A >> B;
            sys.queryCommonFriends(A, B);
        }
    }
    return 0;
}