﻿#include <iostream>
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

struct p_u_gran 
{
    int i_edge;
    int type;
    double p;
    double u;
};

Grid init_grid(string path);

vector<Center> find_center(Grid grid);

vector<double> find_d(vector<Center> center);

vector<conn> find_conn_edge_with_fv(Grid grid);

vector<lenght> find_lenght_edge(Grid grid);

vector<area> find_area_fv(Grid grid, vector<Center> center);

vector<Vertex> system_coord_sdvig_for_naprav(Grid grid, Center center, vector<int> ver);

vector<d_vn> find_d_vn(Grid grid, vector<conn> connection, vector<Center> center);

vector<d_gran> find_d_gran(Grid grid, vector<conn> connection, vector<Center> center);

vector<p_u_gran> find_p_u_gran(Grid grid, string path);

bool srav_angle(Vertex one, Vertex two);

void vivod_vtk(Grid grid, vector<Center> center, vector<double> value);

int main()
{
    string add = "C:\\Stud\\DIPLOM\\2D\\setka.grid";
    double M_g=10000.0;
    Grid grid;
    vector<Center> center;
    //vector<double> d;
    vector<conn> connection;
    vector<lenght> len;
    vector<area> ar;
    vector<d_vn> d_v;
    vector<d_gran> d_g;
    vector<p_u_gran> p_u_gran;
    grid = init_grid(add);
    center = find_center(grid);
    connection = find_conn_edge_with_fv(grid);
    len = find_lenght_edge(grid);
    ar = find_area_fv(grid, center);
    d_v = find_d_vn(grid, connection, center);
    d_g = find_d_gran(grid, connection, center);
    p_u_gran = find_p_u_gran(grid, "C:\\Stud\\DIPLOM\\2D\\gu.txt");

    size_t n = center.size();

    double j_plus = 0.0;
    double j_minus = 0.0;
    double p_plus = 0.0;
    double p_minus = 0.0;
    // Fill M and rhs
    SparseMatrix M(n);
    std::vector<double> rhs(n, 0);
    for (size_t i = 0; i < d_v.size(); ++i)
    {
        M.add(connection[d_v[i].i_edge].fv[0], connection[d_v[i].i_edge].fv[0], (2.0 * M_g * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[1], connection[d_v[i].i_edge].fv[1], (2.0 * M_g * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[0], connection[d_v[i].i_edge].fv[1], -(2.0 * M_g * len[d_v[i].i_edge].l) / d_v[i].d);
        M.add(connection[d_v[i].i_edge].fv[1], connection[d_v[i].i_edge].fv[0], -(2.0 * M_g * len[d_v[i].i_edge].l) / d_v[i].d);
    }
    for (size_t i = 0; i < n; i++)
    {
        M.add(i, i, ar[i].ar * (j_plus + j_minus));
    }

    for (size_t i = 0; i < d_g.size(); i++)
    {
        //M.add(connection[d_g[i].i_edge].fv[0], connection[d_g[i].i_edge].fv[0], (2.0 * M_g * len[d_g[i].i_edge].l) / d_g[i].d);
        if (p_u_gran[d_g[i].i_edge].type == 1)
        {
            //rhs[connection[d_g[i].i_edge].fv[0]] += ar[connection[d_g[i].i_edge].fv[0]].ar * (M_g * len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].p) / d_g[i].d;
            rhs[connection[d_g[i].i_edge].fv[0]] += 2.0 * (M_g * len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].p) / d_g[i].d;
            M.add(connection[d_g[i].i_edge].fv[0], connection[d_g[i].i_edge].fv[0], (2.0 * M_g * len[d_g[i].i_edge].l) / d_g[i].d);
        }
        else if(p_u_gran[d_g[i].i_edge].type == 2)
        {
            //rhs[connection[d_g[i].i_edge].fv[0]] += (-1.0) * ar[connection[d_g[i].i_edge].fv[0]].ar * (len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].u);
            rhs[connection[d_g[i].i_edge].fv[0]] += (-2.0) * (len[d_g[i].i_edge].l * p_u_gran[d_g[i].i_edge].u);
        }
    }
    for (size_t i = 0; i < n; i++)
    {
        rhs[i] += ar[i].ar * (p_plus * j_plus + p_minus * j_minus);
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

    vivod_vtk(grid, center, x);
    int k = 0;
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
                    split(sec_data,line, is_any_of("\t"), token_compress_on);
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
    int n=0;
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
        sort(n_s.begin(), n_s.end(),srav_angle);
        for (size_t p = 0; p < n_s.size(); p++)
        {
            ind[p] = n_s[p].index;
        }
        for (size_t k = 0; k < ind.size() - 1; k++)
        {
            //sum_p += grid.vertex[ind[k]].x * grid.vertex[ind[k + 1]].y;
            //sum_m += grid.vertex[ind[k+1]].x * grid.vertex[ind[k]].y;
            sum_p += n_s[k].x * n_s[k + 1].y;
            sum_m += n_s[k+1].x * n_s[k].y;
        }
        //sum=sum_p + grid.vertex[ind[ind.size() - 1]].x * grid.vertex[ind[0]].y- sum_m- grid.vertex[ind[0]].x * grid.vertex[ind.size() - 1].y;
        sum = sum_p + n_s[ind.size() - 1].x * n_s[0].y - sum_m - n_s[0].x * n_s[ind.size() - 1].y;
        a.i_fv = i;
        a.ar = abs(sum)/2.0;
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
            d_v.d=sqrt(abs(center[connection[i].fv[0]].x - center[connection[i].fv[1]].x) \
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
                d_g.d = abs(A * center[connection[i].fv[1]].x + B * center[connection[i].fv[1]].y + C)/sqrt(A*A+B*B);
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

void vivod_vtk(Grid grid, vector<Center> center, vector<double> value)
{
    vector<string> sec_data;
    ofstream out;          // поток для записи
    vector<int> e;
    vector<int> ind;
    vector<Vertex> n_s;
    //split(sec_data, name, is_any_of("\\"), token_compress_on);
    //out.open(sec_data[sec_data.size() - 1] + ".txt"); // окрываем файл для записи
    out.open("press_fract_value.vtk");
    if (out.is_open())
    {
        out << "# vtk DataFile Version 3.0" << endl;
        out << "fract_cells" << endl;
        out << "ASCII" << endl;
        out << "DATASET UNSTRUCTURED_GRID" << endl;
        out << "POINTS " << grid.vertex.size() << " double" << endl;
        for (size_t i = 0; i < grid.vertex.size(); i++)
        {
            out << grid.vertex[i].x << " " << grid.vertex[i].y << " " << 0.0 << endl;
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
        out << "SCALARS un float 1" << endl;
        out << "LOOKUP_TABLE default" << endl;
        for (size_t i = 0; i < value.size(); i++)
        {
            out << value[i] << endl;
        }
    }

    cout << "End of program" << endl;
}