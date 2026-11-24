// Ядро симуляции: суставы, существа, нейросеть (с отмиранием нейронов), еда, мир, обучение.
// Не зависит от графики (только стандартная библиотека C++17).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace evo {

constexpr double PI = 3.14159265358979323846;
inline double g_rotStiff = 5.0;  // "жёсткость" тела на поворот: чем больше, тем меньше тело вертится при двух опорах

// ----------------------------------------------------------------- математика
struct Vec2 {
    double x = 0, y = 0;
    Vec2() {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2& operator+=(Vec2 o) { x += o.x; y += o.y; return *this; }
    double len() const { return std::sqrt(x * x + y * y); }
    double dot(Vec2 o) const { return x * o.x + y * o.y; }
    double cross(Vec2 o) const { return x * o.y - y * o.x; }
};
inline Vec2 fromAngle(double a) { return {std::cos(a), std::sin(a)}; }
inline double wrapAngle(double a) {
    while (a > PI) a -= 2 * PI;
    while (a < -PI) a += 2 * PI;
    return a;
}
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 1) : s(seed * 2685821657736338717ULL + 1442695040888963407ULL) {
        if (!s) s = 88172645463325252ULL;
        for (int i = 0; i < 8; i++) next();
    }
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * uni(); }
    double gauss() {
        double u = std::max(uni(), 1e-12), v = uni();
        return std::sqrt(-2 * std::log(u)) * std::cos(2 * PI * v);
    }
};

// ----------------------------------------------------------------- нейросеть
// Рекуррентная сеть: входы -> скрытый слой (tanh, с обратной связью) -> выходы (tanh).
// Связь с весом ровно 0 не существует ("отмерла"). Скрытый нейрон с hAlive=0 удалён.
struct Brain;

// Статистика использования связей и нейронов за один раунд (нужна для отмирания).
struct BrainStats {
    std::vector<double> s1, q1, sr, qr, s2, q2, sh, qh;  // суммы и суммы квадратов вкладов
    double n = 0;
    void init(int nIn, int nHid, int nOut) {
        s1.assign(nHid * nIn, 0.0); q1 = s1;
        sr.assign(nHid * nHid, 0.0); qr = sr;
        s2.assign(nOut * nHid, 0.0); q2 = s2;
        sh.assign(nHid, 0.0); qh = sh;
        n = 0;
    }
};

struct PruneResult {
    int links = 0, neurons = 0;
};

struct Brain {
    int nIn = 0, nHid = 0, nOut = 0;
    std::vector<double> w1, wr, b1, w2, b2;  // w1[h][in], wr[h][h'], w2[o][h]
    std::vector<char> hAlive;                // скрытый нейрон жив
    std::vector<uint8_t> k1, kr, k2, kh;     // сколько поколений подряд связь/нейрон был "мало используем"
    std::vector<double> h, out;              // состояние
    // Разреженная копия весов (только ненулевые связи живых нейронов): быстрее, чем перебирать все входы
    bool compiled = false;
    std::vector<int> inS, inI, recS, recI, outS, outI;
    std::vector<double> inW, recW, outW;

    void compile() {
        inS.assign(nHid + 1, 0); recS.assign(nHid + 1, 0); outS.assign(nOut + 1, 0);
        inI.clear(); inW.clear(); recI.clear(); recW.clear(); outI.clear(); outW.clear();
        for (int i = 0; i < nHid; i++) {
            if (hAlive[i]) {
                for (int j = 0; j < nIn; j++) if (w1[i * nIn + j] != 0.0) { inI.push_back(j); inW.push_back(w1[i * nIn + j]); }
                for (int k = 0; k < nHid; k++) if (wr[i * nHid + k] != 0.0) { recI.push_back(k); recW.push_back(wr[i * nHid + k]); }
            }
            inS[i + 1] = (int)inI.size(); recS[i + 1] = (int)recI.size();
        }
        for (int o = 0; o < nOut; o++) {
            for (int i = 0; i < nHid; i++) if (w2[o * nHid + i] != 0.0) { outI.push_back(i); outW.push_back(w2[o * nHid + i]); }
            outS[o + 1] = (int)outI.size();
        }
        compiled = true;
    }

    void init(int i, int hd, int o) {
        nIn = i; nHid = hd; nOut = o;
        w1.assign(hd * i, 0.0); wr.assign(hd * hd, 0.0); b1.assign(hd, 0.0);
        w2.assign(o * hd, 0.0); b2.assign(o, 0.0);
        hAlive.assign(hd, 1);
        k1.assign(hd * i, 0); kr.assign(hd * hd, 0); k2.assign(o * hd, 0); kh.assign(hd, 0);
        h.assign(hd, 0.0); out.assign(o, 0.0);
    }
    double& W1(int hi, int in) { return w1[hi * nIn + in]; }
    double& WR(int hi, int hj) { return wr[hi * nHid + hj]; }
    double& W2(int o, int hi) { return w2[o * nHid + hi]; }

    int aliveNeurons() const {
        int c = 0;
        for (char a : hAlive) c += a;
        return c;
    }
    int aliveLinks() const {
        int c = 0;
        for (double w : w1) c += (w != 0.0);
        for (double w : wr) c += (w != 0.0);
        for (double w : w2) c += (w != 0.0);
        return c;
    }
    int maxLinks() const { return nHid * nIn + nHid * nHid + nOut * nHid; }

    void think(const std::vector<double>& in, BrainStats* st = nullptr) {
        if (!compiled) compile();
        double nh[64];
        for (int i = 0; i < nHid; i++) {
            if (!hAlive[i]) { nh[i] = 0; continue; }
            double s = b1[i];
            for (int k = inS[i]; k < inS[i + 1]; k++) {
                double c = inW[k] * in[inI[k]];
                s += c;
                if (st) { st->s1[i * nIn + inI[k]] += c; st->q1[i * nIn + inI[k]] += c * c; }
            }
            for (int k = recS[i]; k < recS[i + 1]; k++) {
                double c = recW[k] * h[recI[k]];
                s += c;
                if (st) { st->sr[i * nHid + recI[k]] += c; st->qr[i * nHid + recI[k]] += c * c; }
            }
            nh[i] = std::tanh(s);
        }
        for (int i = 0; i < nHid; i++) h[i] = nh[i];
        for (int o = 0; o < nOut; o++) {
            double s = b2[o];
            for (int k = outS[o]; k < outS[o + 1]; k++) {
                double c = outW[k] * h[outI[k]];
                s += c;
                if (st) { st->s2[o * nHid + outI[k]] += c; st->q2[o * nHid + outI[k]] += c * c; }
            }
            out[o] = std::tanh(s);
        }
        if (st) {
            for (int i = 0; i < nHid; i++) { st->sh[i] += h[i]; st->qh[i] += h[i] * h[i]; }
            st->n += 1;
        }
    }

    // Случайные отклонения: существующие веса сдвигаются на N(0, sigma) с вероятностью rate;
    // отсутствующие связи между живыми нейронами изредка "вырастают" (вероятность grow).
    // inOk/outOk - какие входы/выходы вообще есть у тела (неиспользуемые слоты не обрастают связями).
    void mutate(Rng& r, double rate, double sigma, double grow, const std::vector<char>* inOk = nullptr,
                const std::vector<char>* outOk = nullptr) {
        compiled = false;
        auto mutW = [&](double& w, bool live) {
            if (w != 0.0) {
                if (r.uni() < rate) {
                    w = clampd(w + r.gauss() * sigma, -9.0, 9.0);
                    if (w == 0.0) w = 1e-6;
                }
            } else if (live && grow > 0 && r.uni() < grow) {
                w = r.gauss() * 0.3;
                if (w == 0.0) w = 1e-6;
            }
        };
        for (int i = 0; i < nHid; i++) {
            if (!hAlive[i]) continue;
            for (int j = 0; j < nIn; j++) mutW(w1[i * nIn + j], !inOk || (*inOk)[j]);
            for (int k = 0; k < nHid; k++) mutW(wr[i * nHid + k], hAlive[k]);
            if (r.uni() < rate) b1[i] = clampd(b1[i] + r.gauss() * sigma, -9.0, 9.0);
        }
        // слабые связи отмирают (иначе ИИ только разрастается): вес по модулю < 0.03 -> связи нет
        for (double& w : w1) if (w != 0.0 && std::fabs(w) < 0.03) w = 0.0;
        for (double& w : wr) if (w != 0.0 && std::fabs(w) < 0.03) w = 0.0;
        for (double& w : w2) if (w != 0.0 && std::fabs(w) < 0.03) w = 0.0;
        for (int o = 0; o < nOut; o++) {
            bool oOk = !outOk || (*outOk)[o];
            for (int i = 0; i < nHid; i++) mutW(w2[o * nHid + i], hAlive[i] && oOk);
            if (r.uni() < rate) b2[o] = clampd(b2[o] + r.gauss() * sigma, -9.0, 9.0);
        }
    }

    void resizeHidden(int n) {
        Brain nb;
        nb.init(nIn, n, nOut);
        int m = std::min(n, nHid);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < nIn; j++) nb.w1[i * nIn + j] = w1[i * nIn + j];
            for (int k = 0; k < m; k++) nb.wr[i * n + k] = wr[i * nHid + k];
            nb.b1[i] = b1[i]; nb.hAlive[i] = hAlive[i];
        }
        for (int o = 0; o < nOut; o++) {
            for (int i = 0; i < m; i++) nb.w2[o * n + i] = w2[o * nHid + i];
            nb.b2[o] = b2[o];
        }
        for (int i = 0; i < nHid && i < n; i++) nb.h[i] = h[i];
        for (int i = m; i < n; i++) nb.hAlive[i] = 0;
        *this = nb;
    }

    // Новый скрытый нейрон (занимает место отмершего или добавляется в конец): несколько случайных слабых связей.
    bool addNeuron(Rng& r, const std::vector<char>& inOk, const std::vector<char>& outOk, int maxHid = 40) {
        compiled = false;
        int slot = -1;
        for (int i = 0; i < nHid; i++)
            if (!hAlive[i]) { slot = i; break; }
        if (slot < 0) {
            if (nHid >= maxHid) return false;
            resizeHidden(nHid + 1);
            slot = nHid - 1;
        }
        for (int j = 0; j < nIn; j++) w1[slot * nIn + j] = 0;
        for (int k = 0; k < nHid; k++) { wr[slot * nHid + k] = 0; wr[k * nHid + slot] = 0; }
        for (int o = 0; o < nOut; o++) w2[o * nHid + slot] = 0;
        b1[slot] = 0; h[slot] = 0; hAlive[slot] = 1; kh[slot] = 0;
        std::vector<int> ins, outs;
        for (int j = 0; j < nIn; j++) if (inOk[j]) ins.push_back(j);
        for (int o = 0; o < nOut; o++) if (outOk[o]) outs.push_back(o);
        for (int t = 0; t < 3 && !ins.empty(); t++) w1[slot * nIn + ins[(int)(r.uni() * ins.size())]] = r.gauss() * 0.5 + 1e-6;
        for (int t = 0; t < 2 && !outs.empty(); t++) w2[outs[(int)(r.uni() * outs.size())] * nHid + slot] = r.gauss() * 0.2 + 1e-6;
        return true;
    }

    // Отмирание. Связь "используется", если разброс её вклада (w*x) составляет не меньше rel (10%) от
    // самой изменчивой связи того же нейрона-приёмника. Скрытый нейрон "используется", если его активность
    // меняется не меньше чем на rel (std >= 0.1) или он куда-то передаёт сигнал. Если порог не набран
    // need (2) поколения подряд - связь/нейрон удаляется, а его постоянная часть переносится в смещение
    // приёмника (поведение почти не меняется).
    PruneResult prune(const BrainStats& st, double rel = 0.10, int need = 2) {
        PruneResult res;
        compiled = false;
        if (st.n < 30) return res;
        const double n = st.n;
        auto sd = [&](double s, double q) {
            double m = s / n;
            return std::sqrt(std::max(0.0, q / n - m * m));
        };
        // связи, входящие в скрытые нейроны
        for (int i = 0; i < nHid; i++) {
            if (!hAlive[i]) continue;
            double mx = 0;
            for (int j = 0; j < nIn; j++)
                if (w1[i * nIn + j] != 0.0) mx = std::max(mx, sd(st.s1[i * nIn + j], st.q1[i * nIn + j]));
            for (int k = 0; k < nHid; k++)
                if (wr[i * nHid + k] != 0.0) mx = std::max(mx, sd(st.sr[i * nHid + k], st.qr[i * nHid + k]));
            if (mx < 1e-9) continue;  // нейрон получает только константы: решает правило нейрона
            for (int j = 0; j < nIn; j++) {
                int id = i * nIn + j;
                if (w1[id] == 0.0) continue;
                if (sd(st.s1[id], st.q1[id]) < rel * mx) {
                    if (++k1[id] >= need) { b1[i] += st.s1[id] / n; w1[id] = 0; k1[id] = 0; res.links++; }
                } else k1[id] = 0;
            }
            for (int k = 0; k < nHid; k++) {
                int id = i * nHid + k;
                if (wr[id] == 0.0) continue;
                if (sd(st.sr[id], st.qr[id]) < rel * mx) {
                    if (++kr[id] >= need) { b1[i] += st.sr[id] / n; wr[id] = 0; kr[id] = 0; res.links++; }
                } else kr[id] = 0;
            }
        }
        // связи, входящие в выходы
        for (int o = 0; o < nOut; o++) {
            double mx = 0;
            for (int i = 0; i < nHid; i++)
                if (w2[o * nHid + i] != 0.0) mx = std::max(mx, sd(st.s2[o * nHid + i], st.q2[o * nHid + i]));
            if (mx < 1e-9) continue;
            for (int i = 0; i < nHid; i++) {
                int id = o * nHid + i;
                if (w2[id] == 0.0) continue;
                if (sd(st.s2[id], st.q2[id]) < rel * mx) {
                    if (++k2[id] >= need) { b2[o] += st.s2[id] / n; w2[id] = 0; k2[id] = 0; res.links++; }
                } else k2[id] = 0;
            }
        }
        // нейроны
        for (int i = 0; i < nHid; i++) {
            if (!hAlive[i]) continue;
            bool hasOut = false;  // передаёт ли сигнал кому-то (кроме себя)
            for (int o = 0; o < nOut && !hasOut; o++) hasOut = w2[o * nHid + i] != 0.0;
            for (int j = 0; j < nHid && !hasOut; j++) hasOut = (j != i && hAlive[j] && wr[j * nHid + i] != 0.0);
            bool still = sd(st.sh[i], st.qh[i]) < rel;
            if (still) kh[i] = (uint8_t)std::min(250, kh[i] + 1); else kh[i] = 0;
            if (!hasOut || kh[i] >= need) {
                double m = st.sh[i] / n;  // средняя активность - переносится в смещения получателей
                for (int o = 0; o < nOut; o++) {
                    double& w = w2[o * nHid + i];
                    if (w != 0.0) { b2[o] += w * m; w = 0; }
                }
                for (int j = 0; j < nHid; j++) {
                    double& w = wr[j * nHid + i];
                    if (w != 0.0) { if (j != i) b1[j] += w * m; w = 0; }
                }
                for (int j = 0; j < nIn; j++) w1[i * nIn + j] = 0;
                for (int k = 0; k < nHid; k++) wr[i * nHid + k] = 0;
                b1[i] = 0;
                hAlive[i] = 0;
                kh[i] = 0;
                res.neurons++;
            }
        }
        return res;
    }
};

// ----------------------------------------------------------------- гены
// Сустав - отрезок между двумя узлами. parent - узел со стороны корня (точка вращения),
// child - дальний конец. Угол сустава = rest + phi (phi управляется существом).
struct JointGene {
    int parent, child;
    double length;
    double rest;     // угол покоя относительно родителя
    double limit;    // макс. |phi|
    double maxRate;  // макс. скорость поворота, рад/с
};

// Рот и зубы дают энергию только через свой желудок (StomachMouth - для рта, StomachTeeth - для зубов); жир +5 к максимуму энергии.
enum class OrganType { Mouth = 0, Genital = 1, Teeth = 2, StomachMouth = 3, StomachTeeth = 4, Fat = 5 };
constexpr int N_ORGAN_TYPES = 6;
constexpr double FAT_ENERGY = 5.0;       // +к максимуму энергии за каждый орган жира

struct OrganGene {
    OrganType type;
    int node;       // в каком узле (месте соединения суставов) стоит орган
    int refJoint;   // сустав, задающий направление "вперёд"
    double fov;     // угол обзора (рот) (рад)
    double range;   // дальность зрения (рот)
};

constexpr int MAX_J = 24;
constexpr int MAX_N = MAX_J + 1;
constexpr double JOINT_HP = 30.0;

// Входы и выходы нейросети имеют фиксированные "слоты" (по номеру сустава / узла),
// поэтому у существ с разным числом суставов сеть одного размера и мозг можно наследовать.
constexpr int N_SECT = 7;                  // секторов зрения
constexpr int IN_EYE = 0;                  // 0..6   еда в секторах
constexpr int IN_EYEC = N_SECT;            // 7..13  другие существа в секторах
constexpr int IN_CHARGE = 2 * N_SECT;      // 14
constexpr int IN_ENERGY = IN_CHARGE + 1;
constexpr int IN_CLK_S = IN_CHARGE + 2;
constexpr int IN_CLK_C = IN_CHARGE + 3;
constexpr int IN_WANDER = IN_CHARGE + 4;
constexpr int IN_TOUCH = IN_CHARGE + 5;    // столкновение (знак - с какой стороны отодвинуться)
constexpr int IN_HURT = IN_CHARGE + 6;     // самое повреждённое звено (0..1)
constexpr int IN_ANG = IN_CHARGE + 7;      // углы суставов
constexpr int IN_PAIN = IN_ANG + MAX_J;    // боль в суставах (получен урон)
constexpr int IN_PIN = IN_PAIN + MAX_J;    // стопа закреплена (по номеру узла)
constexpr int IN_EYEW = IN_PIN + MAX_N;     // 94..100 стены в секторах зрения
constexpr int N_IN_V7 = IN_EYEW;           // размер входа до появления зрения на стены (старые мозги дополняются нулями)
constexpr int N_IN = IN_EYEW + N_SECT;
constexpr int OUT_PIN = MAX_J;             // выходы: суставы 0..23, затем закрепление стоп по узлам
constexpr int OUT_HEAL = OUT_PIN + MAX_N;
constexpr int OUT_REPRO = OUT_HEAL + 1;
constexpr int N_OUT = OUT_REPRO + 1;
constexpr int N_OUT_DIG = N_OUT + 1;     // у мозгов, сохранённых в v11.11-11.13, был лишний выход (копатель, его больше нет): при загрузке он отбрасывается

struct Genome {
    int nodeCount = 0;
    std::vector<double> nodeRadius;  // визуальный размер точки
    std::vector<JointGene> joints;   // родители раньше детей
    std::vector<OrganGene> organs;
    double clockRate = 5.0;          // частота внутреннего ритма, рад/с
    double wanderRate = 1.1;         // частота "блуждающего" сигнала, рад/с
    Brain brain;                     // веса

    std::vector<int> degrees() const {
        std::vector<int> d(nodeCount, 0);
        for (auto& jt : joints) { d[jt.parent]++; d[jt.child]++; }
        return d;
    }
    // узлы, которые могут закрепляться на земле (одно соединение), по возрастанию номера
    std::vector<int> leafNodes() const {
        std::vector<int> d = degrees(), r;
        for (int n = 1; n < nodeCount; n++)
            if (d[n] == 1) r.push_back(n);
        return r;
    }
    int organAt(int node) const {
        for (size_t i = 0; i < organs.size(); i++)
            if (organs[i].node == node) return (int)i;
        return -1;
    }
    // какие слоты входов/выходов реально есть у этого тела
    void masks(std::vector<char>& inOk, std::vector<char>& outOk) const {
        inOk.assign(N_IN, 0);
        outOk.assign(N_OUT, 0);
        for (int i = 0; i <= IN_HURT; i++) inOk[i] = 1;
        for (int i = IN_EYEW; i < N_IN; i++) inOk[i] = 1;
        for (size_t j = 0; j < joints.size(); j++) {
            inOk[IN_ANG + j] = 1;
            inOk[IN_PAIN + j] = 1;
            outOk[j] = 1;
        }
        for (int n : leafNodes()) { inOk[IN_PIN + n] = 1; outOk[OUT_PIN + n] = 1; }
        outOk[OUT_HEAL] = 1;
        outOk[OUT_REPRO] = 1;
    }
};

inline std::string inputName(int i) {
    static const char* fixed[] = {"food", "energy", "clk sin", "clk cos", "wander", "touch", "hurt"};
    if (i < IN_EYEC) return "eye" + std::to_string(i);
    if (i < IN_CHARGE) return "see" + std::to_string(i - IN_EYEC);
    if (i < IN_ANG) return fixed[i - IN_CHARGE];
    if (i < IN_PAIN) return "ang" + std::to_string(i - IN_ANG);
    if (i < IN_PIN) return "pain" + std::to_string(i - IN_PAIN);
    if (i < IN_EYEW) return "foot" + std::to_string(i - IN_PIN);
    return "wall" + std::to_string(i - IN_EYEW);
}
inline std::string outputName(int o) {
    if (o < OUT_PIN) return "joint" + std::to_string(o);
    if (o < OUT_HEAL) return "pin" + std::to_string(o - OUT_PIN);
    return o == OUT_HEAL ? "heal" : "birth";
}

// ----------------------------------------------------------------- текстовый формат мозга
// Весь мозг записывается одной текстовой строкой, чтобы его можно было скопировать/вставить/прислать.
inline std::string brainToText(const Genome& g) {
    const Brain& b = g.brain;
    std::string s = "EVOBRAIN 1\n";
    char buf[64];
    std::sn
