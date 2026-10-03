#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

#define lr 0.0005
#define h 1.e-10
#define tol 1.e-5

vector<vector<double>> train_data; 
vector<double> test_actual;
vector<double> train_raw;

double min_price, max_price;

double a1 = 0.1, a2 = 0.1, a3 = 0.1, b = 0.1;
double dEa1 = 1.0, dEa2 = 1.0, dEa3 = 1.0, dEb = 1.0;

double norm(double p)   { return (p - min_price) / (max_price - min_price); }
double denorm(double p) { return p * (max_price - min_price) + min_price; }

vector<double> read_price_csv(const string& filename) {
    vector<double> prices;
    ifstream file(filename);
    string line;
    getline(file, line);
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string idx, date, day, price_str;
        getline(ss, idx, ',');
        getline(ss, date, ',');
        getline(ss, day, ',');
        getline(ss, price_str, ',');
        prices.push_back(stod(price_str));
    }
    return prices;
}

double y_(int i) {
    return a1 * train_data[i][0] + a2 * train_data[i][1] + a3 * train_data[i][2] + b;
}

double E() {
    double galat = 0.0;
    int indeks = 0;
    for (vector<double>& nilai : train_data) {
        galat += pow(nilai[3] - y_(indeks), 2);
        indeks++;
    }
    return galat;
}

void grad() {
    double temp = a1;
    double a, bb;

    a1 += h;
    a = E();
    a1 -= 2 * h;
    bb = E();
    dEa1 = (a - bb) / (2 * h);
    a1 = temp;

    temp = a2;
    a2 += h;
    a = E();
    a2 -= 2 * h;
    bb = E();
    dEa2 = (a - bb) / (2 * h);
    a2 = temp;

    temp = a3;
    a3 += h;
    a = E();
    a3 -= 2 * h;
    bb = E();
    dEa3 = (a - bb) / (2 * h);
    a3 = temp;
    
    temp = b;
    b += h;
    a = E();
    b -= 2 * h;
    bb = E();
    dEb = (a - bb) / (2 * h);
    b = temp;
}

int main() {
    train_raw   = read_price_csv("train_jan_agu_2026.csv");
    test_actual = read_price_csv("test_sep_2026.csv");

    if (train_raw.empty() || test_actual.empty()) {
        cerr << "Error: File CSV tidak ditemukan!\n";
        return 1;
    }

    min_price = train_raw[0];
    max_price = train_raw[0];
    for (double p : train_raw) {
        if (p < min_price) min_price = p;
        if (p > max_price) max_price = p;
    }

    for (size_t i = 3; i < train_raw.size(); ++i) {
        train_data.push_back({
            norm(train_raw[i - 3]),
            norm(train_raw[i - 2]),
            norm(train_raw[i - 1]),
            norm(train_raw[i])
        });
    }

    ofstream loss_out("history_loss_linear.csv");
    loss_out << "iteration,loss,grad_norm\n";

    int count = 0;
    int max_iter = 3000;

    cout << "Memulai training...\n";
    while (sqrt(pow(dEa1, 2) + pow(dEa2, 2) + pow(dEa3, 2) + pow(dEb, 2)) > tol && count < max_iter) {
        count++;
        grad();

        a1 -= dEa1 * lr;
        a2 -= dEa2 * lr;
        a3 -= dEa3 * lr;
        b -= dEb * lr;

        double current_loss = E();
        double g_norm = sqrt(pow(dEa1, 2) + pow(dEa2, 2) + pow(dEa3, 2) + pow(dEb, 2));
        loss_out << count << "," << current_loss << "," << g_norm << "\n";

        if (count % 200 == 0 || count == 1) {
            cout << "Iter: " << count << " | Loss: " << current_loss << " | |Grad|: " << g_norm << endl;
        }
    }
    loss_out.close();

    cout << "Selesai! a1=" << a1 << " a2=" << a2 << " a3=" << a3 << " b=" << b << " E=" << E() << endl;

    ofstream pred_out("predictions_sep_linear.csv");
    pred_out << "date,day_name,actual_price_idr,pred_walkforward_idr,pred_rollout_idr\n";

    vector<double> full_norm;
    for (double p : train_raw) full_norm.push_back(norm(p));
    for (double p : test_actual) full_norm.push_back(norm(p));

    double r1 = norm(train_raw[train_raw.size() - 3]);
    double r2 = norm(train_raw[train_raw.size() - 2]);
    double r3 = norm(train_raw[train_raw.size() - 1]);

    for (size_t i = 0; i < test_actual.size(); ++i) {
        size_t idx = train_raw.size() + i;
        double wf_norm = a1 * full_norm[idx - 3] + a2 * full_norm[idx - 2] + a3 * full_norm[idx - 1] + b;

        double ro_norm = a1 * r1 + a2 * r2 + a3 * r3 + b; 
        r1 = r2, r2 = r3, r3 = ro_norm;

        pred_out << "2026-09-" << (i + 1 < 10 ? "0" : "") << (i + 1) << ",Day,"
                 << (long long)test_actual[i] << ","
                 << (long long)denorm(wf_norm) << ","
                 << (long long)denorm(ro_norm) << "\n";
    }
    pred_out.close();

    cout << "Hasil tersimpan di predictions_sep_linear.csv dan history_loss_linear.csv\n";
    return 0;
}
