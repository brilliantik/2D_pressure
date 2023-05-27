#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

#include <boost/algorithm/string.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>

#include <J_plus_minus.h>

#include <ctime>

using namespace std;
using namespace boost;
using namespace algorithm;

// ====================== CSRMatrix
struct CSRMatrix {
    std::vector<size_t> ptr;
    std::vector<size_t> col;
    std::vector<double> val;
};

// ======================= SparseMatrix
class SparseMatrix {
public:
    typedef std::map<size_t, double> row_t;

    // Constructs empty NxN matrix
    SparseMatrix(size_t N);

    // Returns number of rows
    size_t dim() const;
    // Returns number of non-zero entries
    size_t n_non_zero() const;

    // Returns value at (irw, icol) position
    double value(size_t irow, size_t icol) const;
    // Returns read-only irow-th row as {column: value} map.
    const row_t& row(size_t irow) const;

    // Clears irow-th row and sets unity to its diagonal position
    void unit_row(size_t irow);

    // M[irow, icol] += val
    void add(size_t irow, size_t icol, double val);

    // M[irow, icol] = 0
    void rem(size_t irow, size_t icol);

    // solves M*x = rhs using amgcl
    // returns <number of iterations, final tolerance> tuple
    std::tuple<size_t, double> solve(const std::vector<double>& rhs, std::vector<double>& x) const;

    // Performs matrix-vector multiplication in form:
    //     ret += coef * M*x
    void mult_vec(const std::vector<double>& x, double coef, std::vector<double>& ret) const;

    // Converts matrix to csr format
    CSRMatrix as_csr() const;
    // Converts matrix to dense format
    std::vector<double> as_dense() const;

    // Sets maximum allowed iterations and minimum allowed tolerance for the amgcl solver
    void set_solver_params(size_t maxit, double tolerance);
private:
    const size_t N;
    std::vector<row_t> data;
    size_t solver_maximum_iterations = 1000;
    double solver_tolerance = 1e-12;
};

// ======================== SparseMatrix implementation
std::ostream& operator<<(std::ostream& os, const SparseMatrix& M) {
    if (M.dim() <= 12) {
        // use dense output for small matrices
        std::vector<double> dt = M.as_dense();
        double* it = dt.data();
        for (size_t irow = 0; irow < M.dim(); ++irow) {
            for (size_t icol = 0; icol < M.dim(); ++icol) {
                os << std::setw(10) << *it++ << " ";
            }
            os << std::endl;
        }
    }
    else {
        // prints only non-zero entries for large matrices
        for (size_t irow = 0; irow < M.dim(); ++irow) {
            os << "--- ROW " << irow << std::endl;
            os << "/";
            for (auto& it : M.row(irow)) {
                os << " " << it.first << ": " << it.second << " /";
            }
            os << std::endl;
        }
    }
    return os;
}

SparseMatrix::SparseMatrix(size_t N) : N(N), data(N) { }

size_t SparseMatrix::dim() const {
    return N;
}
const SparseMatrix::row_t& SparseMatrix::row(size_t irow) const {
    return data[irow];
}

double SparseMatrix::value(size_t irow, size_t icol) const {
    auto fnd = data[irow].find(icol);
    if (fnd == data[irow].end()) {
        return 0;
    }
    else {
        return fnd->second;
    }
}

size_t SparseMatrix::n_non_zero() const {
    size_t ret = 0;
    for (auto& row : data) ret += row.size();
    return ret;
}
void SparseMatrix::unit_row(size_t irow) {
    data[irow].clear();
    data[irow].emplace(irow, 1);
}

void SparseMatrix::add(size_t irow, size_t icol, double val) {
    row_t& row = data[irow];
    auto r = row.emplace(icol, val);
    if (!r.second) r.first->second += val;
}

void SparseMatrix::rem(size_t irow, size_t icol) {
    row_t& row = data[irow];
    auto r = row.find(icol);
    if (r != row.end()) {
        row.erase(r);
    }
}

void SparseMatrix::mult_vec(const std::vector<double>& x, double coef, std::vector<double>& ret) const {
    ret.resize(dim(), 0);
    for (size_t irow = 0; irow < dim(); ++irow) {
        for (auto& it : data[irow]) {
            ret[irow] += coef * it.second * x[it.first];
        }
    }
}

void SparseMatrix::set_solver_params(size_t maxit, double tolerance) {
    solver_maximum_iterations = maxit;
    solver_tolerance = tolerance;
}

CSRMatrix SparseMatrix::as_csr() const {
    CSRMatrix ret;
    size_t nn = n_non_zero();

    ret.ptr.resize(dim() + 1, 0);
    ret.col.resize(nn);
    ret.val.resize(nn);

    size_t k = 0;
    for (size_t irow = 0; irow < dim(); ++irow) {
        for (auto& it : data[irow]) {
            ret.col[k] = it.first;
            ret.val[k] = it.second;
            ++k;
        }
        ret.ptr[irow + 1] = k;
    }

    return ret;
}

std::vector<double> SparseMatrix::as_dense() const {
    std::vector<double> ret(N * N, 0);
    for (size_t irow = 0; irow < N; ++irow) {
        const row_t& row = data[irow];
        for (auto& it : row) {
            size_t gind = N * irow + it.first;
            ret[gind] = it.second;
        }
    }
    return ret;
}
std::tuple<size_t, double> SparseMatrix::solve(const std::vector<double>& rhs, std::vector<double>& x) const {
    CSRMatrix csr = as_csr();

    amgcl::backend::crs<double, size_t> amgcl_matrix;
    amgcl_matrix.own_data = false;
    amgcl_matrix.nrows = amgcl_matrix.ncols = dim();
    amgcl_matrix.nnz = n_non_zero();
    amgcl_matrix.ptr = csr.ptr.data();
    amgcl_matrix.col = csr.col.data();
    amgcl_matrix.val = csr.val.data();

    boost::property_tree::ptree prm;
    prm.put("solver.type", "fgmres");
    prm.put("solver.tol", solver_tolerance);
    prm.put("solver.maxiter", solver_maximum_iterations);
    prm.put("precond.coarsening.type", "smoothed_aggregation");
    prm.put("precond.relax.type", "spai0");

    using backend_t = amgcl::backend::builtin<double>;
    using solver_t = amgcl::make_solver<
        amgcl::amg<
        backend_t,
        amgcl::runtime::coarsening::wrapper,
        amgcl::runtime::relaxation::wrapper
        >,
        amgcl::runtime::solver::wrapper<backend_t>
    >;
    solver_t slv(amgcl_matrix, prm);

    x.resize(dim());
    auto ret = slv(rhs, x);

    if (std::get<0>(ret) >= solver_maximum_iterations) {
        std::ostringstream os;
        os << "WARNING: Sparse matrix solution failed to converge with tolerance: "
            << std::scientific << std::setprecision(2) << std::get<1>(ret)
            << std::endl;
        std::cout << os.str();
    }

    return ret;
}


struct Vertex
{
    double x;
    double y;
    double z;
    int index;
    double angle = atan2(y, x);
};

struct Edge
{
    int v1;
    int v2;
    int index;
};

struct FV
{
    int n;
    vector<int> edge;
    int index;
    double M;//безразмерная гидропроводность
};

struct Grid
{
    vector<Vertex> vertex;
    vector<Edge> edge;
    vector<FV> fv;
};

struct Center
{
    double x;
    double y;
    int index_fv;
};

struct conn
{
    int i_edge;
    //int n;
    vector<int> fv;
};

struct lenght
{
    int i_edge;
    double l;
};

struct area
{
    int i_fv;
    double ar;
};

struct d_vn
{
    int i_edge;
    double d;
};

struct d_gran
{
    int i_edge;
    double d;
};

struct gran
{
    int i_edge;
    int type;
};

struct p_u_gran
{
    int i_edge;
    int type;
    double p;
    double u;
};

struct Cell
{
    int n;
    vector<int> v;
    int index;
};

struct Data
{
    double data;
    int index;
};

struct MM//безразмерная гидропроводность
{
    int i_edge;
    double value;
};

Grid init_grid(string path);

Grid init_grid_vtk(string path);

vector<Center> find_center(Grid grid);

vector<double> find_d(vector<Center> center);

vector<conn> find_conn_edge_with_fv(Grid grid);

vector<lenght> find_lenght_edge(Grid grid);

vector<area> find_area_fv(Grid grid, vector<Center> center);

vector<Vertex> system_coord_sdvig_for_naprav(Grid grid, Center center, vector<int> ver);

vector<d_vn> find_d_vn(Grid grid, vector<conn> connection, vector<Center> center);

vector<d_gran> find_d_gran(Grid grid, vector<conn> connection, vector<Center> center);

vector<p_u_gran> find_p_u_gran(Grid grid, string path);

vector<p_u_gran> find_p_u_gran(Grid grid, string path);

vector<MM> find_M_edge(Grid grid, vector<d_vn> d_v, vector<d_gran> d_g, vector<conn> connection, vector<Center> center);

bool srav_angle(Vertex one, Vertex two);

bool srav_e(Edge one, Edge two);

bool srav_i(gran one, gran two);

bool srav_i_M(MM one, MM two);

void vivod_vtk(Grid grid, vector<Center> center, vector<double> value, string name);

void create_gu_for_vtk(Grid grid, vector<d_gran> d_g, vector<d_vn> d_v);

vector<double> init_data(string path, int n);

void vivod_xyz(Grid grid, vector<Center> center, vector<double> value, string name);

void vivod_profile_un(Grid grid, vector<Center> center, vector<conn> connection, vector<double> un, string name);
void vivod_profile_oblako(Grid grid, vector<Center> center, vector<conn> connection, vector<double> un, string name);

double d_xt(double d, double R, double H, double L, int scen);

//double R = 100.0;
//double H_minus = 25.0;
//double H_plus = 25.0;

double x_max = -999999.0;
double x_min = 999999.0;
double y_max = -999999.0;
double y_min = 999999.0;

int S_up_plus = -999999;
int S_down_plus = -999999;
int S_up_minus = -999999;
int S_down_minus = -999999;

double R_up_plus = -999999.0;
double R_down_plus = -999999.0;
double R_up_minus = -999999.0;
double R_down_minus = -999999.0;

double L_up_plus = -999999.0;
double L_down_plus = -999999.0;
double L_up_minus = -999999.0;
double L_down_minus = -999999.0;

double H_up = -999999.0; //plus
double H_down = -999999.0; //minus

double delta_f = -999999.0;//полураскрытие трещины размерное
double H_char = -999999.0;//характерная длина трещины размерная
double mu = -999999.0;
double kr = -999999.0;
double dp = -999999.0;

double (*j_minus)(double x, double R, double H, double L);
double (*j_plus)(double x, double R, double H, double L);
double (*j)(double x, double R, double H, double L);

double P_G = -99999999999999.0;
double P_f_start = -99999999999999.0;
double P_well = -99999999999999.0;

int main()
{
     unsigned int start_time = clock(); // начальное время
     //S_up_plus = 2;
     //S_down_plus = 2;
     //S_up_minus = 2;
     //S_down_minus = 2;

     //R_up_plus = 1.375;
     //R_down_plus = 1.375;
     //R_up_minus = 1.375;
     //R_down_minus = 1.375;

     //L_up_plus = 0.5;
     //L_down_plus = 0.5;
     //L_up_minus = 0.5;
     //L_down_minus = 0.5;

     S_up_plus = 1;
     S_down_plus = 1;
     S_up_minus = 1;
     S_down_minus = 1;

     R_up_plus = 8.0;
     R_down_plus = 8.0;
     R_up_minus = 8.0;
     R_down_minus = 8.0;

     L_up_plus = -999999.0;
     L_down_plus = -999999.0;
     L_up_minus = -999999.0;
     L_down_minus = -999999.0;

     delta_f = (1.0 / 100) / 2.0;
     H_char = 100.0;
     mu = 1.0 / 1000.0;
     kr = 1.0 / 1000000000000.0;
     dp = 1013250.0;

     P_G = 1.0;
     P_f_start = P_G;
     P_well = 0.0;

    string add = "C:\\Stud\\DIPLOM\\2D\\setka.grid";
    //double M_g = 10.0;
    vector<double> M_fv;
    vector<MM> M_M;
    Grid grid;
    vector<Center> center;
    //vector<double> d;
    vector<conn> connection;
    vector<lenght> len;
    vector<area> ar;
    vector<d_vn> d_v;
    vector<d_gran> d_g;
    vector<p_u_gran> p_u_gran;
    //grid = init_grid(add);
    //grid = init_grid_vtk("D:\\Stud\\DIPLOM\\2D\\press_fract_value_test.vtk");

    string path_perm = "C:\\Stud\\DIPLOM\\2D\\start_files\\perm_fract_0_0.00000000.vtk";
    string path_press = "C:\\Stud\\DIPLOM\\2D\\start_files\\press_fract_0_0.00000000.vtk";
    string path_un_plus = "C:\\Stud\\DIPLOM\\2D\\start_files\\un_fract_0_0.00000000_dim.vtk";
    string path_un_minus = "C:\\Stud\\DIPLOM\\2D\\start_files\\un_fract_0_0.00000000_dim.vtk";

    grid = init_grid_vtk(path_perm);

    vector<double> press_nast(grid.fv.size());
    vector<double> un_nast_plus(grid.fv.size());
    vector<double> un_nast_minus(grid.fv.size());

    //grid = init_grid_vtk("C:\\Stud\\DIPLOM\\2D\\perm_fract_0_0.00000000.vtk");
    for (size_t i = 0; i < grid.fv.size(); i++)
    {
        M_fv.push_back(grid.fv[i].M);
        M_fv[i] = 1000.0;
        grid.fv[i].M = 1000.0;
        //if (grid.fv[i].index == 415 || grid.fv[i].index == 400)
        //{
        //    M_fv[i] = 1.0/10000.0;
        //    grid.fv[i].M = 1.0/10000.0;
        //}
        //else
        //{
        //    M_fv[i] = 1.0;
        //    grid.fv[i].M = 1.0;
        //}
    }
    center = find_center(grid);

    connection = find_conn_edge_with_fv(grid);
    len = find_lenght_edge(grid);
    ar = find_area_fv(grid, center);
    d_v = find_d_vn(grid, connection, center);
    d_g = find_d_gran(grid, connection, center);

    M_M = find_M_edge(grid, d_v, d_g, connection, center);


    create_gu_for_vtk(grid, d_g, d_v);
    //p_u_gran = find_p_u_gran(grid, "D:\\Stud\\DIPLOM\\2D\\gu_vtk.txt");
    p_u_gran = find_p_u_gran(grid, "C:\\Stud\\DIPLOM\\2D\\2D_pressure\\gu_vtk_test.txt");

    size_t n = center.size();

    //double j_plus = 0.0;
    //double j_minus = 0.0;
    double p_plus = P_G;
    double p_minus = P_G;
    // Fill M and rhs
    SparseMatrix M(n);

    std::vector<double> rhs(n, 0);
    //std::vector<double> rhs(n, P_f_start);//начальное давление на трещине не работает при p_well=0 p_g=1

    for (size_t i = 0; i < d_v.size(); ++i)
    {
        M.add(connection[d_v[i].i_edge].fv[0], connection[d_v[i].i_edge].fv[0], (2.0 * M_M[d_v[i].i_edge].value * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[1], connection[d_v[i].i_edge].fv[1], (2.0 * M_M[d_v[i].i_edge].value * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[0], connection[d_v[i].i_edge].fv[1], -(2.0 * M_M[d_v[i].i_edge].value * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[1], connection[d_v[i].i_edge].fv[0], -(2.0 * M_M[d_v[i].i_edge].value * len[d_v[i].i_edge].l) / d_v[i].d);
    }
    for (size_t i = 0; i < n; i++)
    {
        if (center[i].x > 0.0)
        {
            switch (S_up_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_up_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            M.add(i, i, ar[i].ar * (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up)));
        }
        else if (center[i].x < 0.0)
        {
            switch (S_down_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_down_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            M.add(i, i, ar[i].ar * (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down)));
        }
        else
        {
            cout << "Error switch j_plus/j_minus in M" << endl;
        }
        //M.add(i, i, ar[i].ar * (j_plus + j_minus));
    }

    for (size_t i = 0; i < d_g.size(); i++)
    {
        if (p_u_gran[d_g[i].i_edge].type == 1)
        {
            rhs[connection[d_g[i].i_edge].fv[0]] += 2.0 * (M_M[d_g[i].i_edge].value * len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].p) / d_g[i].d;
            M.add(connection[d_g[i].i_edge].fv[0], connection[d_g[i].i_edge].fv[0], (2.0 * M_M[d_g[i].i_edge].value * len[d_g[i].i_edge].l) / d_g[i].d);
        }
        else if (p_u_gran[d_g[i].i_edge].type == 2)
        {
            rhs[connection[d_g[i].i_edge].fv[0]] += (-2.0) * (len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].u);
        }
    }
    for (size_t i = 0; i < n; i++)
    {
        if (center[i].x > 0.0)
        {
            switch (S_up_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_up_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            rhs[i] += ar[i].ar * (p_plus * j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + p_minus * j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
        }
        else if (center[i].x < 0.0)
        {
            switch (S_down_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_down_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            rhs[i] += ar[i].ar * (p_plus * j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + p_minus * j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
        }
        else
        {
            cout << "Error switch j_plus/j_minus in rhs" << endl;
        }
        //rhs[i] += ar[i].ar * (p_plus * j_plus + p_minus * j_minus);
    }

    // Print M and rhs
    std::cout << "========== Matrix" << std::endl;
    std::cout << M << std::endl;
    std::cout << "========== Right hand side" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cout << rhs[i] << " ";
    }
    std::cout << std::endl;

    // Solve SLE
    std::vector<double> x(n);
    std::tuple<size_t, double> r = M.solve(rhs, x);

    // Print solution
    std::cout << "========== Solution" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cout << "x_y[" << center[i].x << " , " << center[i].y << "] = " << x[i] << " " << std::endl;
    }
    std::cout << std::endl;

    // Print additional solution info
    std::cout << "========== Solution properties" << std::endl;
    std::cout << "iterations: " << std::get<0>(r) << std::endl;
    std::cout << "tolerance : " << std::get<1>(r) << std::endl;

    vector<double> u_n(x.size());
    vector<double> u_n_plus(x.size());
    vector<double> u_n_minus(x.size());
    vector<double> j(x.size());
    
    //С учетом параметра d для КО у торца если у торца все КО одинаковой прямоугольной геометрии
    double x_center_max = -999999.0;
    int i_center_max = -999999;
    double x_center_min = 999999.0;
    int i_center_min = -999999;
    for (size_t i = 0; i < center.size(); i++)
    {
        if (center[i].x > x_center_max)
        {
            x_center_max = center[i].x;
            i_center_max = i;
        }
        if (center[i].x < x_center_min)
        {
            x_center_min = center[i].x;
            i_center_min = i;
        }
    }
    double d_max = -999999.0;
    double d_min = -999999.0;
    //С поправкой
    for (size_t i = 0; i < grid.fv[i_center_max].n; i++)
    {
        if (grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v1].x == x_max && grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v2].x == x_max)
        {
            continue;
        }
        else if (grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v1].x == x_max && grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v2].x < x_max)
        {
            d_max = len[grid.fv[i_center_max].edge[i]].l;
        }
        else if (grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v1].x < x_max && grid.vertex[grid.edge[grid.fv[i_center_max].edge[i]].v2].x == x_max)
        {
            d_max = len[grid.fv[i_center_max].edge[i]].l;
        }
    }
    for (size_t i = 0; i < grid.fv[i_center_min].n; i++)
    {
        if (grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v1].x == x_min && grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v2].x == x_min)
        {
            continue;
        }
        else if (grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v1].x == x_min && grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v2].x > x_min)
        {
            d_min = len[grid.fv[i_center_min].edge[i]].l;
        }
        else if (grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v1].x > x_min && grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v2].x == x_min)
        {
            d_min = len[grid.fv[i_center_min].edge[i]].l;
        }
        //if (grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v1].x == x_min || grid.vertex[grid.edge[grid.fv[i_center_min].edge[i]].v2].x == x_min)
        //{
        //    d_min = len[grid.fv[i_center_min].edge[i]].l;
        //}
    }

    for (size_t i = 0; i < x.size(); i++)
    {
        if (center[i].x > 0.0)
        {
            switch (S_up_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_up_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            if (center[i].x == x_center_max)
            {
                for (size_t k = 0; k < grid.fv[i].n; k++)
                {
                    if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x == x_max && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x == x_max)
                    {
                        continue;
                    }
                    else if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x == x_max && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x < x_max)
                    {
                        d_max = len[grid.fv[i].edge[i]].l;
                    }
                    else if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x < x_max && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x == x_max)
                    {
                        d_max = len[grid.fv[i].edge[k]].l;
                    }
                }
                u_n_plus[i] = (P_G - x[i]) * (j_plus(d_xt(d_max / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up, S_up_plus) / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up));
                u_n_minus[i] = (P_G - x[i]) * (j_minus(d_xt(d_max / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up, S_up_minus) / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
                j[i] = (j_plus(d_xt(d_max / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up, S_up_plus) / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(d_xt(d_max / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up, S_up_minus) / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
                u_n[i] = (P_G - x[i]) * (j_plus(d_xt(d_max / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up, S_up_plus) / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(d_xt(d_max / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up, S_up_minus) / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
                d_max = -999999.0;
                d_min = -999999.0;
            }
            else 
            {
                u_n_plus[i] = (P_G - x[i]) * (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up));
                u_n_minus[i] = (P_G - x[i]) * (j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
                j[i] = (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
                u_n[i] = (P_G - x[i]) * (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
            }
        }
        else if (center[i].x < 0.0)
        {
            switch (S_down_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_down_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            if (center[i].x == x_center_min)
            {
                for (size_t k = 0; k < grid.fv[i].n; k++)
                {
                    if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x == x_min && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x == x_min)
                    {
                        continue;
                    }
                    else if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x == x_min && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x > x_min)
                    {
                        d_min = len[grid.fv[i].edge[k]].l;
                    }
                    else if (grid.vertex[grid.edge[grid.fv[i].edge[k]].v1].x > x_min && grid.vertex[grid.edge[grid.fv[i].edge[k]].v2].x == x_min)
                    {
                        d_min = len[grid.fv[i].edge[k]].l;
                    }
                }
                u_n_plus[i] = (P_G - x[i]) * (j_plus(abs(d_xt(d_min / H_down, R_down_plus / H_down, 1.0, L_down_plus / H_down, S_down_plus) / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down));
                u_n_minus[i] = (P_G - x[i]) * (j_minus(abs(d_xt(d_min / H_down, R_down_minus / H_down, 1.0, L_down_minus / H_down, S_down_minus) / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
                j[i] = (j_plus(abs(d_xt(d_min / H_down, R_down_plus / H_down, 1.0, L_down_plus / H_down, S_down_plus) / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(d_xt(d_min / H_down, R_down_minus / H_down, 1.0, L_down_minus / H_down, S_down_minus) / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
                u_n[i] = (P_G - x[i]) * (j_plus(abs(d_xt(d_min / H_down, R_down_plus / H_down, 1.0, L_down_plus / H_down, S_down_plus) / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(d_xt(d_min / H_down, R_down_minus / H_down, 1.0, L_down_minus / H_down, S_down_minus) / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
                d_max = -999999.0;
                d_min = -999999.0;
            }
            else
            {
                u_n_plus[i] = (P_G - x[i]) * (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down));
                u_n_minus[i] = (P_G - x[i]) * (j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
                j[i] = (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
                u_n[i] = (P_G - x[i]) * (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
            }
        }
    }

    //Без поправки
    
    /*for (size_t i = 0; i < x.size(); i++)
    {
        if (center[i].x > 0.0)
        {
            switch (S_up_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_up_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            u_n_plus[i] = (P_G - x[i]) * (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up));
            u_n_minus[i] = (P_G - x[i]) * (j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
            j[i] = (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
            u_n[i] = (P_G - x[i]) * (j_plus(center[i].x / H_up, R_up_plus / H_up, 1.0, L_up_plus / H_up) + j_minus(center[i].x / H_up, R_up_minus / H_up, 1.0, L_up_minus / H_up));
        }
        else if (center[i].x < 0.0)
        {
            switch (S_down_plus)
            {
            case 1:
            {
                j_plus = j1;
                break;
            }
            case 2:
            {
                j_plus = j2;
                break;
            }
            }
            switch (S_down_minus)
            {
            case 1:
            {
                j_minus = j1;
                break;
            }
            case 2:
            {
                j_minus = j2;
                break;
            }
            }
            u_n_plus[i] = (P_G - x[i]) * (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down));
            u_n_minus[i] = (P_G - x[i]) * (j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
            j[i] = (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
            u_n[i]=(P_G - x[i]) * (j_plus(abs(center[i].x / H_down), R_down_plus / H_down, 1.0, L_down_plus / H_down) + j_minus(abs(center[i].x / H_down), R_down_minus / H_down, 1.0, L_down_minus / H_down));
        }
    }*/

    //vivod_vtk(grid, center, x, "press_fract_value");
    //vivod_vtk(grid, center, M_fv, "perm_fract_value");
    //vivod_vtk(grid, center, u_n, "un_fract_value");
    //vivod_vtk(grid, center, j, "j+-_fract_value");

    press_nast = init_data(path_press, ar.size());
    un_nast_plus = init_data(path_un_plus, ar.size());
    un_nast_minus = init_data(path_un_minus, ar.size());

    vector<double> R_press(ar.size());
    vector<double> R_un_plus(ar.size());
    vector<double> R_un_minus(ar.size());
    double Q = 0.0;

    for (size_t i = 0; i < ar.size(); i++)
    {
        R_press[i] = (x[i] - press_nast[i]) / (P_G - P_well);
        R_un_plus[i] = (u_n_plus[i] * delta_f / H_char - un_nast_plus[i] * delta_f * mu / (kr * dp)) / (un_nast_plus[i] * delta_f * mu / (kr * dp));
        R_un_minus[i] = (u_n_minus[i] * delta_f / H_char - un_nast_minus[i] * delta_f * mu / (kr * dp)) / (un_nast_minus[i] * delta_f * mu / (kr * dp));
        //R_un[i] = (u_n[i] - un_nast[i]/2.0) / (un_nast[i]/2.0);
        Q += (delta_f / H_char) * ar[i].ar * (u_n_plus[i] + u_n_minus[i]);
    }

    //vivod_vtk(grid, center, x, "press_value");
    //vivod_vtk(grid, center, R_press, "R_press_value");
    //vivod_vtk(grid, center, u_n_plus, "un_plus_value");
    vivod_profile_oblako(grid, center, connection, u_n_plus, "profile_un_plus_oblako");
    vivod_profile_un(grid, center, connection, u_n_plus, "profile_un_plus");
    //vivod_vtk(grid, center, R_un_plus, "R_un_plus_value");
    //vivod_vtk(grid, center, R_un_minus, "R_un_minus_value");
    //vivod_xyz(grid, center, R_un_minus, "R_un_minus_value");
    //vivod_xyz(grid, center, x, "press_value");
    //vivod_xyz(grid, center, u_n_plus, "un_plus_value");

    //vivod_profile_un(grid, center, connection, u_n_plus, "profile_un_plus");
    //vivod_profile_oblako(grid, center, connection, u_n_plus, "profile_un_plus_oblako");
    //vivod_profile_un(grid, center, connection, u_n_minus, "profile_un_minus");
    //vivod_profile_un(grid, center, connection, u_n, "profile_un");

    cout << "Q = " << Q << endl;
    unsigned int end_time = clock(); // конечное время
    unsigned int search_time = end_time - start_time; // искомое время
    cout << "Time = " << search_time / 1000.0 << endl;
    int k = 0;
}

vector<MM> find_M_edge(Grid grid, vector<d_vn> d_v, vector<d_gran> d_g, vector<conn> connection, vector<Center> center)
{
    vector<MM> M_M;
    MM m;
    double k1= -999999.0;
    double k2= -999999.0;
    double d1= -999999.0;
    double d2= -999999.0;
    double edge_center_x = -999999.0;
    double edge_center_y = -999999.0;
    for (size_t i = 0; i < d_v.size(); i++)
    {
        k1 = grid.fv[connection[d_v[i].i_edge].fv[0]].M;
        k2 = grid.fv[connection[d_v[i].i_edge].fv[1]].M;
        edge_center_x = (grid.vertex[grid.edge[d_v[i].i_edge].v1].x + grid.vertex[grid.edge[d_v[i].i_edge].v2].x) / (2.0);
        edge_center_y = (grid.vertex[grid.edge[d_v[i].i_edge].v1].y + grid.vertex[grid.edge[d_v[i].i_edge].v2].y) / (2.0);
        d1 = (center[connection[d_v[i].i_edge].fv[0]].x - edge_center_x) \
            * (center[connection[d_v[i].i_edge].fv[0]].x - edge_center_x) \
            + (center[connection[d_v[i].i_edge].fv[0]].y - edge_center_y) \
            * (center[connection[d_v[i].i_edge].fv[0]].y - edge_center_y);
        d2 = (center[connection[d_v[i].i_edge].fv[1]].x - edge_center_x) \
            * (center[connection[d_v[i].i_edge].fv[1]].x - edge_center_x) \
            + (center[connection[d_v[i].i_edge].fv[1]].y - edge_center_y) \
            * (center[connection[d_v[i].i_edge].fv[1]].y - edge_center_y);
        m.i_edge = d_v[i].i_edge;
        m.value = (k1 * k2 * (d1 + d2)) / (k2 * d1 + k1 * d2);
        M_M.push_back(m);
    }

    for (size_t i = 0; i < d_g.size(); i++)
    {
        m.i_edge = d_g[i].i_edge;
        if (connection[d_g[i].i_edge].fv[0] != -1 && connection[d_g[i].i_edge].fv[1] == -1)
        {
            m.value= grid.fv[connection[d_g[i].i_edge].fv[0]].M;
        }
        else if (connection[d_g[i].i_edge].fv[0] == -1 && connection[d_g[i].i_edge].fv[1] != -1)
        {
            m.value = grid.fv[connection[d_g[i].i_edge].fv[1]].M;
        }
        else
        {
            cout << "Error in find M for gran edge" << endl;
        }
        M_M.push_back(m);
    }
    sort(M_M.begin(), M_M.end(), srav_i_M);
    return M_M;
}

void create_gu_for_vtk(Grid grid, vector<d_gran> d_g, vector<d_vn> d_v)
{
    gran gr;
    vector<gran> g;
    for (size_t i = 0; i < d_v.size(); i++)
    {
        gr.i_edge = d_v[i].i_edge;
        gr.type = 0;
        g.push_back(gr);
    }
    for (size_t i = 0; i < d_g.size(); i++)
    {
        gr.i_edge = d_g[i].i_edge;
        if ((grid.vertex[grid.edge[gr.i_edge].v1].x == x_max) && (grid.vertex[grid.edge[gr.i_edge].v2].x == x_max))
        {
            gr.type = 5;
        }
        else if ((grid.vertex[grid.edge[gr.i_edge].v1].x == x_min) && (grid.vertex[grid.edge[gr.i_edge].v2].x == x_min))
        {
            gr.type = 4;
        }
        else if ((grid.vertex[grid.edge[gr.i_edge].v1].y == y_max) && (grid.vertex[grid.edge[gr.i_edge].v2].y == y_max))
        {
            gr.type = 2;
        }
        else if ((grid.vertex[grid.edge[gr.i_edge].v1].y == y_min) && (grid.vertex[grid.edge[gr.i_edge].v2].y == y_min))
        {
            gr.type = 3;
        }
        else
        {
            gr.type = 1;
        }
        g.push_back(gr);
    }
    sort(g.begin(), g.end(), srav_i);//уникальное значение в векторе
    ofstream out;          // поток для записи
    out.open("gu_vtk_test.txt");
    if (out.is_open())
    {
        out << "j\tType\tv1\tv2" << endl;
        
        for (size_t i = 0; i < g.size(); i++)
        {
            if (g[i].type == 1)
            {
                out << g[i].i_edge << "\t" << "1" << "\t" << to_string(P_well) << "\t" << "0" << "\t" << endl;
                //out << g[i].i_edge << "\t" << "1" << "\t" << "0" << "\t" << "0" << "\t" << endl;//Почему не работает с давление на скважине = 0
            }
            else if (g[i].type == 0)
            {
                out << g[i].i_edge << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << endl;
            }
            else
            {
                out << g[i].i_edge << "\t" << "2" << "\t" << "0" << "\t" << "0" << "\t" << endl;
                //out << g[i].i_edge << "\t" << "1" << "\t" << "1" << "\t" << "0" << "\t" << endl;///ТЕстовая задача для проверки
            }
        }
        //for (size_t i = 0; i < grid.edge.size(); i++)
        //{
        //    out << grid.edge[i].index << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << endl;
        //}
    }
    cout << "Create gu for vtk" << endl;
}

Grid init_grid_vtk(string path)
{
    Grid grid;
    string line;
    Vertex v;
    Cell cell;
    Edge ed;
    Edge ed_2;
    vector<Cell> c;
    vector<Edge> edg;
    vector<Edge> edg_2;
    FV f_v;
    vector<FV> f;
    Data d;
    typedef vector< string > split_vector_type;
    split_vector_type sec_data;
    vector<Vertex> ver;
    double x_norm = -999999.0;
    double y_norm = -999999.0;

    split(sec_data, line, is_any_of("\t"), token_compress_on);

    ifstream in(path);
    if (in.is_open())
    {
        while (getline(in, line))
        {
            if (line == "DATASET UNSTRUCTURED_GRID")
            {
                getline(in, line);
                split(sec_data, line, is_any_of(" "), token_compress_on);
                int n_points = stoi(sec_data[1]);
                sec_data.clear();
                for (size_t i = 0; i < n_points; i++)
                {
                    getline(in, line);
                    split(sec_data, line, is_any_of(" "), token_compress_on);
                    v.x = stod(sec_data[0]);//+ нормировка R/H для
                    //v.y = stod(sec_data[1]);
                    //v.z = stod(sec_data[2]);
                    v.y = stod(sec_data[2]);//потому что у марселя перевернутая
                    v.z = stod(sec_data[1]);//потому что у марселя перевернутая
                    v.index = i;
                    grid.vertex.push_back(v);
                    sec_data.clear();
                }
                getline(in, line);
                split(sec_data, line, is_any_of(" "), token_compress_on);
                int n_cells = stoi(sec_data[1]);
                sec_data.clear();
                for (size_t i = 0; i < n_cells; i++)
                {
                    getline(in, line);
                    split(sec_data, line, is_any_of(" \t"), token_compress_on);
                    cell.n = stoi(sec_data[0]);
                    cell.index = i;
                    for (size_t j = 0; j < cell.n; j++)
                    {
                        cell.v.push_back(stoi(sec_data[j + 1]));
                    }
                    c.push_back(cell);
                    cell.v.clear();
                }
                for (size_t i = 0; i < c.size(); i++)
                {
                    for (size_t j = 0; j < c[i].v.size()-1; j++)
                    {
                        if (c[i].v[j] < c[i].v[j + 1])
                        {
                            ed.v1 = c[i].v[j];
                            ed.v2 = c[i].v[j + 1];
                            ed.index = -999999;

                            ed_2.v1 = c[i].v[j];
                            ed_2.v2 = c[i].v[j + 1];
                            ed_2.index = i;
                        }
                        else if (c[i].v[j] > c[i].v[j + 1])
                        {
                            ed.v1 = c[i].v[j + 1];
                            ed.v2 = c[i].v[j];
                            ed.index = -999999;

                            ed_2.v1 = c[i].v[j + 1];
                            ed_2.v2 = c[i].v[j];
                            ed_2.index = i;
                        }
                        edg.push_back(ed);
                        edg_2.push_back(ed_2);
                    }
                    if (c[i].v[0] < c[i].v[c[i].v.size() - 1])
                    {
                        ed.v1 = c[i].v[0];
                        ed.v2 = c[i].v[c[i].v.size() - 1];
                        ed.index = -999999;

                        ed_2.v1 = c[i].v[0];
                        ed_2.v2 = c[i].v[c[i].v.size() - 1];
                        ed_2.index = i;
                    }
                    else if (c[i].v[0] > c[i].v[c[i].v.size() - 1])
                    {
                        ed.v1 = c[i].v[c[i].v.size() - 1];
                        ed.v2 = c[i].v[0];
                        ed.index = -999999;

                        ed_2.v1 = c[i].v[c[i].v.size() - 1];
                        ed_2.v2 = c[i].v[0];
                        ed_2.index = i;
                    }
                    edg.push_back(ed);
                    edg_2.push_back(ed_2);
                }
                sort(edg.begin(), edg.end(), srav_e);//уникальное значение в векторе
                vector<string> s_srav;
                for (size_t i = 0; i < edg.size(); i++)
                {
                    s_srav.push_back(to_string(edg[i].v1) + "\t" + to_string(edg[i].v2));
                }
                s_srav.erase(unique(s_srav.begin(), s_srav.end()), s_srav.end());//уникальное значение в векторе
                edg.clear();
                for (size_t i = 0; i < s_srav.size(); i++)
                {
                    split(sec_data, s_srav[i], is_any_of("\t"), token_compress_on);
                    ed.v1 = stoi(sec_data[0]);
                    ed.v2 = stoi(sec_data[1]);
                    ed.index = i;
                    edg.push_back(ed);
                }
                grid.edge = edg;
                //edg.clear();

                for (size_t i = 0; i < edg.size(); i++)
                {
                    ver.push_back(grid.vertex[edg[i].v1]);
                    ver.push_back(grid.vertex[edg[i].v2]);
                }
                
                for (size_t i = 0; i < ver.size(); i++)
                {
                    if (ver[i].x > x_max)
                    {
                        x_max = ver[i].x;
                    }
                    if (ver[i].x < x_min)
                    {
                        x_min = ver[i].x;
                    }
                    if (ver[i].y > y_max)
                    {
                        y_max = ver[i].y;
                    }
                    if (ver[i].y < y_min)
                    {
                        y_min = ver[i].y;
                    }
                }
                if (abs(x_max) >= abs(x_min))
                {
                    x_norm = abs(x_max);
                }
                else if (abs(x_max) < abs(x_min))
                {
                    x_norm = abs(x_min);
                }
                for (size_t i = 0; i < grid.vertex.size(); i++)
                {
                    grid.vertex[i].x = grid.vertex[i].x / x_norm;
                    grid.vertex[i].y = grid.vertex[i].y / x_norm;
                }
                x_max = x_max / x_norm;
                x_min = x_min / x_norm;
                y_max = y_max / x_norm;
                y_min = y_min / x_norm;
                
                H_up = abs(x_max);
                H_down = abs(x_min);

                for (size_t i = 0; i < c.size(); i++)
                {
                    f_v.n = c[i].n;
                    f_v.index = c[i].index;
                    for (size_t j = 0; j < edg_2.size(); j++)
                    {
                        if (edg_2[j].index == c[i].index)
                        {
                            for (size_t ii = 0; ii < edg.size(); ii++)
                            {
                                if (edg_2[j].v1 == edg[ii].v1 && edg_2[j].v2 == edg[ii].v2)
                                {
                                    f_v.edge.push_back(edg[ii].index);
                                }
                            }
                        }
                    }
                    grid.fv.push_back(f_v);
                    f_v.edge.clear();
                }
                getline(in, line);
                for (size_t i = 0; i < grid.fv.size(); i++)
                {
                    getline(in, line);
                }
                getline(in, line);
                getline(in, line);
                getline(in, line);
                for (size_t i = 0; i < grid.fv.size(); i++)
                {
                    getline(in, line);
                    //grid.fv[i].M = stod(line);
                    grid.fv[i].M = stod(line) * delta_f / (H_char);//безразмерная гидропроводность
                }
            }
        }
    }
    return grid;
}

Grid init_grid(string path)
{
    Grid grid;
    string line;
    Vertex v;
    Edge e;
    FV f;
    vector<int> ed;
    typedef vector< string > split_vector_type;
    split_vector_type sec_data;

    ifstream in(path);
    if (in.is_open())
    {
        while (getline(in, line))
        {
            if (line == "vertexs")
            {
                getline(in, line);
                while (line != "edges")
                {
                    getline(in, line);
                    if (line == "edges")
                    {
                        sec_data.clear();
                        break;
                    }
                    split(sec_data, line, is_any_of("\t"), token_compress_on);
                    v.x = stod(sec_data[0]);
                    v.y = stod(sec_data[1]);
                    v.index = stoi(sec_data[2]);
                    grid.vertex.push_back(v);
                    sec_data.clear();
                }
                getline(in, line);
                while (line != "FV")
                {
                    getline(in, line);
                    if (line == "FV")
                    {
                        sec_data.clear();
                        break;
                    }
                    split(sec_data, line, is_any_of("\t"), token_compress_on);
                    e.v1 = stoi(sec_data[0]);
                    e.v2 = stoi(sec_data[1]);
                    e.index = stoi(sec_data[2]);
                    grid.edge.push_back(e);
                    sec_data.clear();
                }

                getline(in, line);
                while (!in.eof())
                {
                    getline(in, line);
                    split(sec_data, line, is_any_of("\t"), token_compress_on);
                    if (sec_data[0] == "")
                    {
                        sec_data.clear();
                        ed.clear();
                        break;
                    }
                    f.n = stoi(sec_data[0]);
                    for (size_t i = 0; i < f.n; i++)
                    {
                        ed.push_back(stoi(sec_data[i + 1]));
                    }
                    f.edge = ed;
                    f.index = stoi(sec_data.back());
                    grid.fv.push_back(f);
                    ed.clear();
                    sec_data.clear();
                }
            }
        }
    }
    return grid;
}

vector<Center> find_center(Grid grid)
{
    vector<Center> center;
    Center cen;
    double sum_x = 0.0;
    double sum_y = 0.0;
    vector<int> e;
    vector<Vertex> v;
    vector<int> ind;
    for (size_t i = 0; i < grid.fv.size(); i++)
    {
        for (size_t j = 0; j < grid.fv[i].n; j++)
        {
            e.push_back(grid.fv[i].edge[j]);
            ind.push_back(grid.vertex[grid.edge[e[j]].v1].index);
            ind.push_back(grid.vertex[grid.edge[e[j]].v2].index);
            //v.push_back(grid.vertex[grid.edge[e[j]].v1]);
            //v.push_back(grid.vertex[grid.edge[e[j]].v2]);
        }
        sort(ind.begin(), ind.end());//уникальное значение в векторе
        ind.erase(unique(ind.begin(), ind.end()), ind.end());//уникальное значение в векторе
        for (size_t k = 0; k < ind.size(); k++)
        {
            sum_x += grid.vertex[ind[k]].x;
            sum_y += grid.vertex[ind[k]].y;
        }
        cen.x = sum_x / ind.size();
        cen.y = sum_y / ind.size();
        cen.index_fv = i;
        center.push_back(cen);
        sum_x = 0.0;
        sum_y = 0.0;
        e.clear();
        ind.clear();
        int k = 0;
    }
    return center;
}

vector<double> find_d(vector<Center> center)
{
    vector<double> d;
    return d;
}

vector<conn> find_conn_edge_with_fv(Grid grid)
{
    vector<conn> connection;
    conn connect;
    vector<int> fv;
    int n = 0;
    for (size_t i = 0; i < grid.edge.size(); i++)
    {
        for (size_t j = 0; j < grid.fv.size(); j++)
        {
            for (size_t k = 0; k < grid.fv[j].n; k++)
            {
                if (grid.fv[j].edge[k] == i)
                {
                    //n += 1;
                    fv.push_back(j);
                }
            }
        }
        if (fv.size() == 1)
        {
            fv.push_back(-1);
        }
        connect.i_edge = i;
        //connect.n = n;
        connect.fv = fv;
        connection.push_back(connect);
        fv.clear();
        n = 0;
    }
    return connection;
}

vector<lenght> find_lenght_edge(Grid grid)
{
    vector<lenght> len;
    lenght le;
    for (size_t i = 0; i < grid.edge.size(); i++)
    {
        le.i_edge = i;
        le.l = sqrt(abs(grid.vertex[grid.edge[i].v1].x - grid.vertex[grid.edge[i].v2].x) \
            * abs(grid.vertex[grid.edge[i].v1].x - grid.vertex[grid.edge[i].v2].x) \
            + abs(grid.vertex[grid.edge[i].v1].y - grid.vertex[grid.edge[i].v2].y) \
            * abs(grid.vertex[grid.edge[i].v1].y - grid.vertex[grid.edge[i].v2].y));
        len.push_back(le);
    }
    return len;
}

vector<area> find_area_fv(Grid grid, vector<Center> center)
{
    vector<area> are;
    area a;
    double sum_p = 0.0;
    double sum_m = 0.0;
    double sum = 0.0;
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    for (size_t i = 0; i < grid.fv.size(); i++)
    {
        for (size_t j = 0; j < grid.fv[i].n; j++)
        {
            e.push_back(grid.fv[i].edge[j]);
            ind.push_back(grid.vertex[grid.edge[e[j]].v1].index);
            ind.push_back(grid.vertex[grid.edge[e[j]].v2].index);
            //v.push_back(grid.vertex[grid.edge[e[j]].v1]);
            //v.push_back(grid.vertex[grid.edge[e[j]].v2]);
        }
        sort(ind.begin(), ind.end());//уникальное значение в векторе
        ind.erase(unique(ind.begin(), ind.end()), ind.end());//уникальное значение в векторе
        n_s = system_coord_sdvig_for_naprav(grid, center[i], ind);
        sort(n_s.begin(), n_s.end(), srav_angle);
        for (size_t p = 0; p < n_s.size(); p++)
        {
            ind[p] = n_s[p].index;
        }
        for (size_t k = 0; k < ind.size() - 1; k++)
        {
            //sum_p += grid.vertex[ind[k]].x * grid.vertex[ind[k + 1]].y;
            //sum_m += grid.vertex[ind[k+1]].x * grid.vertex[ind[k]].y;
            sum_p += n_s[k].x * n_s[k + 1].y;
            sum_m += n_s[k + 1].x * n_s[k].y;
        }
        //sum=sum_p + grid.vertex[ind[ind.size() - 1]].x * grid.vertex[ind[0]].y- sum_m- grid.vertex[ind[0]].x * grid.vertex[ind.size() - 1].y;
        sum = sum_p + n_s[ind.size() - 1].x * n_s[0].y - sum_m - n_s[0].x * n_s[ind.size() - 1].y;
        a.i_fv = i;
        a.ar = abs(sum) / 2.0;
        are.push_back(a);
        sum_p = 0.0;
        sum_m = 0.0;
        ind.clear();
        n_s.clear();
        e.clear();
        int k = 0;
    }
    return are;
}

vector<Vertex> system_coord_sdvig_for_naprav(Grid grid, Center center, vector<int> ver)
{
    vector<Vertex> new_system;
    Vertex p;
    for (size_t i = 0; i < ver.size(); i++)
    {
        p.x = grid.vertex[ver[i]].x - center.x;
        p.y = grid.vertex[ver[i]].y - center.y;
        p.index = ver[i];
        p.angle = atan2(p.y, p.x);
        new_system.push_back(p);
    }
    return new_system;
    new_system.clear();
}

bool srav_angle(Vertex one, Vertex two)
{
    if (one.angle < two.angle)
    {
        return false;
    }
    else if (one.angle > two.angle)
    {
        return true;
    }
}

bool srav_e(Edge one, Edge two)
{
    if (one.v1 < two.v1)
    {
        return true;
    }
    else if (one.v1 > two.v1)
    {
        return false;
    }
    else if (one.v1 == two.v1)
    {
        if (one.v2 < two.v2)
        {
            return true;
        }
        else if (one.v2 > two.v2)
        {
            return false;
        }
        else if (one.v2 == two.v2)
        {
            return false;
        }
    }
}

bool srav_i(gran one, gran two)
{
    if (one.i_edge < two.i_edge)
    {
        return true;
    }
    else if (one.i_edge > two.i_edge)
    {
        return false;
    }
    else if (one.i_edge == two.i_edge)
    {
        return false;
    }
}

bool srav_i_M(MM one, MM two)
{
    if (one.i_edge < two.i_edge)
    {
        return true;
    }
    else if (one.i_edge > two.i_edge)
    {
        return false;
    }
    else if (one.i_edge == two.i_edge)
    {
        return false;
    }
}

vector<d_vn> find_d_vn(Grid grid, vector<conn> connection, vector<Center> center)
{
    vector<d_vn> dij;
    d_vn d_v;
    double A;
    double B;
    double C;

    for (size_t i = 0; i < connection.size(); i++)
    {
        if (connection[i].fv[0] != -1 && connection[i].fv[1] != -1)
        {
            d_v.i_edge = i;
            d_v.d = sqrt(abs(center[connection[i].fv[0]].x - center[connection[i].fv[1]].x) \
                * abs(center[connection[i].fv[0]].x - center[connection[i].fv[1]].x) \
                + abs(center[connection[i].fv[0]].y - center[connection[i].fv[1]].y) \
                * abs(center[connection[i].fv[0]].y - center[connection[i].fv[1]].y));
            dij.push_back(d_v);
        }
    }
    return dij;
}

vector<d_gran> find_d_gran(Grid grid, vector<conn> connection, vector<Center> center)
{
    vector<d_gran> dij;
    d_gran d_g;
    double A;
    double B;
    double C;
    for (size_t i = 0; i < connection.size(); i++)
    {
        if (connection[i].fv[0] == -1 || connection[i].fv[1] == -1)
        {
            A = grid.vertex[grid.edge[connection[i].i_edge].v2].y \
                - grid.vertex[grid.edge[connection[i].i_edge].v1].y;
            B = grid.vertex[grid.edge[connection[i].i_edge].v1].x \
                - grid.vertex[grid.edge[connection[i].i_edge].v2].x;
            C = grid.vertex[grid.edge[connection[i].i_edge].v1].y \
                * grid.vertex[grid.edge[connection[i].i_edge].v2].x \
                - grid.vertex[grid.edge[connection[i].i_edge].v1].x \
                * grid.vertex[grid.edge[connection[i].i_edge].v2].y;
            d_g.i_edge = i;
            if (connection[i].fv[0] == -1)
            {
                d_g.d = abs(A * center[connection[i].fv[1]].x + B * center[connection[i].fv[1]].y + C) / sqrt(A * A + B * B);
            }
            else if (connection[i].fv[1] == -1)
            {
                d_g.d = abs(A * center[connection[i].fv[0]].x + B * center[connection[i].fv[0]].y + C) / sqrt(A * A + B * B);
            }
            dij.push_back(d_g);
        }
    }
    return dij;
}

vector<p_u_gran> find_p_u_gran(Grid grid, string path)
{
    vector<p_u_gran> p_u_gr;
    p_u_gran p_u_g;
    string line;
    typedef vector< string > split_vector_type;
    split_vector_type sec_data;

    ifstream in(path);
    if (in.is_open())
    {
        getline(in, line);
        while (getline(in, line))
        {
            split(sec_data, line, is_any_of("\t"), token_compress_on);
            if (sec_data[1] == "0")
            {
                p_u_g.i_edge = stoi(sec_data[0]);
                p_u_g.type = 0;
                p_u_g.p = -999999.0;
                p_u_g.u = -999999.0;
            }
            else if (sec_data[1] == "1")
            {
                p_u_g.i_edge = stoi(sec_data[0]);
                p_u_g.type = 1;
                p_u_g.p = stod(sec_data[2]);
                p_u_g.u = -999999.0;
            }
            else if (sec_data[1] == "2")
            {
                p_u_g.i_edge = stoi(sec_data[0]);
                p_u_g.type = 2;
                p_u_g.p = -999999.0;
                p_u_g.u = stod(sec_data[2]);
            }
            p_u_gr.push_back(p_u_g);
        }
    }
    return p_u_gr;
}

void vivod_vtk(Grid grid, vector<Center> center, vector<double> value, string name)
{
    vector<string> sec_data;
    ofstream out;          // поток для записи
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    //split(sec_data, name, is_any_of("\\"), token_compress_on);
    //out.open(sec_data[sec_data.size() - 1] + ".txt"); // окрываем файл для записи
    out.open(name+".vtk");
    if (out.is_open())
    {
        out << "# vtk DataFile Version 3.0" << endl;
        out << "fract_cells" << endl;
        out << "ASCII" << endl;
        out << "DATASET UNSTRUCTURED_GRID" << endl;
        out << "POINTS " << grid.vertex.size() << " double" << endl;
        for (size_t i = 0; i < grid.vertex.size(); i++)
        {
            //out << grid.vertex[i].x << " " << grid.vertex[i].y << " " << 0.0 << endl;
            out << grid.vertex[i].x << " " << 0.0 << " " << grid.vertex[i].y << endl;//у марселя перевернутое
        }
        int size = 0;
        for (size_t i = 0; i < grid.fv.size(); i++)
        {
            size += grid.fv[i].n + 1;
        }
        out << "CELLS " << grid.fv.size() << " " << size << endl;
        vector<int> e;
        vector<int> ind;
        vector<Vertex> n_s;
        for (size_t i = 0; i < grid.fv.size(); i++)
        {
            for (size_t j = 0; j < grid.fv[i].n; j++)
            {
                e.push_back(grid.fv[i].edge[j]);
                ind.push_back(grid.vertex[grid.edge[e[j]].v1].index);
                ind.push_back(grid.vertex[grid.edge[e[j]].v2].index);
            }
            sort(ind.begin(), ind.end());//уникальное значение в векторе
            ind.erase(unique(ind.begin(), ind.end()), ind.end());//уникальное значение в векторе
            n_s = system_coord_sdvig_for_naprav(grid, center[i], ind);
            sort(n_s.begin(), n_s.end(), srav_angle);
            for (size_t p = 0; p < n_s.size(); p++)
            {
                ind[p] = n_s[p].index;
            }
            out << ind.size() << "\t";
            for (size_t ii = 0; ii < ind.size(); ii++)
            {
                out << ind[ii] << " ";
            }
            out << endl;
            int kkk = 0;
            ind.clear();
            n_s.clear();
            e.clear();
        }
        out << "CELL_TYPES" << " " << grid.fv.size() << endl;
        for (size_t i = 0; i < grid.fv.size(); i++)
        {
            out << "7" << endl;
        }
        out << "CELL_DATA" << " " << grid.fv.size() << endl;
        out << "SCALARS " + name + " float 1" << endl;
        out << "LOOKUP_TABLE default" << endl;
        for (size_t i = 0; i < value.size(); i++)
        {
            out << value[i] << endl;
        }
    }

    cout << name + ".vtk" << "...created" << endl;
}

vector<double> init_data(string path, int n)
{
    vector<double> data;
    string line;
    typedef vector< string > split_vector_type;
    split_vector_type sec_data;

    split(sec_data, line, is_any_of("\t"), token_compress_on);

    ifstream in(path);
    if (in.is_open())
    {
        while (getline(in, line))
        {
            while(line!="LOOKUP_TABLE default")
            {
                getline(in, line);
            }
            for (size_t i = 0; i < n; i++)
            {
                getline(in, line);
                data.push_back(stod(line));
            }
        }
    }
    return data;
    data.clear();
}

void vivod_xyz(Grid grid, vector<Center> center, vector<double> value, string name)
{
    vector<string> sec_data;
    ofstream out;          // поток для записи
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    //split(sec_data, name, is_any_of("\\"), token_compress_on);
    //out.open(sec_data[sec_data.size() - 1] + ".txt"); // окрываем файл для записи
    out.open(name + ".txt");
    if (out.is_open())
    {
        for (size_t i = 0; i < center.size(); i++)
        {
            out << center[i].x << "\t" << center[i].y << "\t" << value[i] << endl;
        }
    }

    cout << name + ".txt" << "...created" << endl;
}

double d_xt(double d, double R, double H, double L, int scen)
{
    double x = -999999.0;
    switch (scen)
    {
    case 1:
    {
        j = j1;
        break;
    }
    case 2:
    {
        j = j2;
        break;
    }
    }

    double a = H - d;
    double b = H;
    double value;
    double sum = 0.0;

    int N = 10 + 1;
    double hh =d/(N*1.0);
 
    vector<double> M(N);
    for (int i = 0; i < N; i++)
    {
        M[i] = a + i * hh;
    }
    for (int i = 1; i < N; i++)
    {
        sum += (b - a) / (6.0 * (N - 1.0)) * (j(M[i - 1], R, H, L) + 4.0 * j((M[i] + M[i - 1]) / (2.0), R, H, L) + j(M[i], R, H, L));
    }
    value = sum;
    M.clear();

    int iter = 0;
    double eps = 0.000001;
    double left = a;
    double right = b;
    double fl, fr, f;
    do {
        x = (left + right) / 2.0;
        f = d * j(x, R, H, L) - value;
        if (f > 0) right = x;
        else left = x;
        iter++;
    } while (fabs(f) > eps && iter < 20000);
    return x;
}

void vivod_profile_un(Grid grid, vector<Center> center, vector<conn> connection, vector<double> un, string name)
{
    vector<string> sec_data;
    ofstream out;          // поток для записи
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    vector<Edge> edj;
    for (size_t i = 0; i < grid.edge.size(); i++)
    {
        if (grid.vertex[grid.edge[i].v1].y == 0 && grid.vertex[grid.edge[i].v2].y == 0)
        {
            edj.push_back(grid.edge[i]);
        }
    }
    vector<double> unn(edj.size());
    vector<double> xx(edj.size());
    for (size_t i = 0; i < edj.size(); i++)
    {
        unn[i] = (un[connection[edj[i].index].fv[0]] + un[connection[edj[i].index].fv[1]])/2.0;
        xx[i] = (grid.vertex[edj[i].v1].x + grid.vertex[edj[i].v2].x) / 2.0;
    }

    out.open(name + ".txt");
    if (out.is_open())
    {
        for (size_t i = 0; i < xx.size(); i++)
        {
            out << xx[i] << "\t" << unn[i] << endl;
        }
    }

    cout << name + ".txt" << "...created" << endl;
}

void vivod_profile_oblako(Grid grid, vector<Center> center, vector<conn> connection, vector<double> un, string name)
{
    vector<string> sec_data;
    ofstream out;          // поток для записи
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    vector<Edge> edj;

    out.open(name + ".txt");
    if (out.is_open())
    {
        for (size_t i = 0; i < un.size(); i++)
        {
            out << center[grid.fv[i].index].x << "\t" << un[i] << endl;
        }
    }

    cout << name + ".txt" << "...created" << endl;
}