#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

#define lr 0.0001
#define h 1.e-10
#define tol 1.e-5

vector<vector<double>> train_data; 
vector<double> test_actual;
vector<double> train_raw;

double min_price, max_price;

double w[13];
double b[5];

double dEw[13];
double dEb[5];

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

void init_parameters() {
    for (int i = 1; i <= 12; ++i) {
        w[i] = 0.1 * ((i % 3) + 1);
        dEw[i] = 1.0;
    }
    for (int i = 1; i <= 4; ++i) {
        b[i] = 0.1;
        dEb[i] = 1.0;
    }
}

double forward_nn(double u1, double u2, double u3) {
    double z1 = u1 * w[1] + u2 * w[4] + u3 * w[7] + b[1];
    double z2 = u1 * w[2] + u2 * w[5] + u3 * w[8] + b[2];
    double z3 = u1 * w[3] + u2 * w[6] + u3 * w[9] + b[3];
    double y_hat = z1 * w[10] + z2 * w[11] + z3 * w[12] + b[4];
    return y_hat;
}

double y_(int i) {
    return forward_nn(train_data[i][0], train_data[i][1], train_data[i][2]);
}

double E() {
    double error_total = 0.0;
    for (size_t i = 0; i < train_data.size(); i++) {
        error_total += pow(y_(i) - train_data[i][3], 2);
    }
    return error_total;
}

void grad() {
    for (int i = 1; i <= 12; i++) {
        double temp = w[i];
        w[i] += h;
        double a = E();

        w[i] -= 2 * h;
        double bb = E();

        dEw[i] = (a - bb) / (2 * h);
        w[i] = temp;
    }

    for (int i = 1; i <= 4; i++) {
        double temp = b[i];
        b[i] += h;
        double a = E();

        b[i] -= 2 * h;
        double bb = E();

        dEb[i] = (a - bb) / (2 * h);
        b[i] = temp;
    }
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

    init_parameters();

    ofstream loss_out("history_loss_nn.csv");
    loss_out << "iteration,loss,grad_norm\n";

    int count = 0;
    int max_iter = 3000;

    cout << "Memulai training Neural Network (16 parameter)...\n";

    auto calc_gnorm = [&]() {
        double sq = 0.0;
        for (int i = 1; i <= 12; ++i) sq += pow(dEw[i], 2);
        for (int i = 1; i <= 4; ++i)  sq += pow(dEb[i], 2);
        return sqrt(sq);
    };

    while (calc_gnorm() > tol && count < max_iter) {
        count++;
        grad();

        for (int i = 1; i <= 12; i++) {
            w[i] -= dEw[i] * lr;
        }

        for (int i = 1; i <= 4; i++) {
            b[i] -= dEb[i] * lr;
        }

        double current_loss = E();
        double g_norm = calc_gnorm();
        loss_out << count << "," << current_loss << "," << g_norm << "\n";

        if (count % 200 == 0 || count == 1) {
            cout << "Iter: " << count << " | Loss: " << current_loss << " | |Grad|: " << g_norm << endl;
        }
    }
    loss_out.close();

    cout << "Selesai training NN! E=" << E() << endl;

    ofstream pred_out("predictions_sep_nn.csv");
    pred_out << "date,day_name,actual_price_idr,pred_walkforward_idr,pred_rollout_idr\n";

    vector<double> full_norm;
    for (double p : train_raw) full_norm.push_back(norm(p));
    for (double p : test_actual) full_norm.push_back(norm(p));

    double r1 = norm(train_raw[train_raw.size() - 3]);
    double r2 = norm(train_raw[train_raw.size() - 2]);
    double r3 = norm(train_raw[train_raw.size() - 1]);

    for (size_t i = 0; i < test_actual.size(); ++i) {
        size_t idx = train_raw.size() + i;
        double wf_norm = forward_nn(full_norm[idx - 3], full_norm[idx - 2], full_norm[idx - 1]);

        double ro_norm = forward_nn(r1, r2, r3);
        r1 = r2;
        r2 = r3;
        r3 = ro_norm;

        pred_out << "2026-09-" << (i + 1 < 10 ? "0" : "") << (i + 1) << ",Day,"
                 << (long long)test_actual[i] << ","
                 << (long long)denorm(wf_norm) << ","
                 << (long long)denorm(ro_norm) << "\n";
    }
    pred_out.close();

    cout << "Hasil tersimpan di predictions_sep_nn.csv dan history_loss_nn.csv\n";
    return 0;
}
