#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <memory>
#include <random>

using namespace std;


// 损失函数基类
class LossFunction
{
public:
    LossFunction(double weight) : weight_(weight)
    {}

    virtual void computerJacobian(double u_x) = 0;  // 纯虚函数

    double weight_;
    double jacobians_;

};

// 柯西损失函数
class CauchyLoss : public LossFunction
{
public:
    CauchyLoss(double weight) : LossFunction(weight)
    {}

    void computerJacobian(double u_x) override
    {
        jacobians_ = weight_ * u_x / (1.0 + u_x * u_x);  // 柯西函数对变量的求导
    }
};


// 顶点类
class Vertex
{
public:
    Vertex(int dim) : dimension_(dim)
    {
        ordering_id_ = 0;
        data_.resize(dimension_);
    }

    Eigen::VectorXd data_; // 顶点存储的数据
    int dimension_;        // 顶点维度
    ulong ordering_id_;    // 当前顶点的序号
};

// 约束边类
class Factor
{
public:
    Factor() : loss_function_(nullptr)
    {}

    void setParametersDims(int dim_residual, vector<std::shared_ptr<Vertex>>& vertex)
    {
        dim_resiual_ = dim_residual;
        vertices_ = vertex;
        error_.resize(dim_resiual_);
        jacobians_.resize(vertex.size());
    }
    virtual void computerError() = 0;    // 纯虚数，子类中实现
    virtual void computerJacobian() = 0;

    void setInformationMatrix(Eigen::MatrixXd information)
    {
        information_ = information;
    }



    void setLossFunction(std::shared_ptr<LossFunction> loss_function)
    {
        loss_function_ = loss_function;
    }

    
    vector<std::shared_ptr<Vertex>> vertices_;
    std::shared_ptr<LossFunction> loss_function_;
    Eigen::VectorXd error_;  // 误差项
    double squared_error_; // 平方误差项，便于分解计算
    vector<Eigen::MatrixXd> jacobians_; // 误差关于优化变量的雅克比矩阵

    int dim_resiual_;

    Eigen::VectorXd means_;
    Eigen::MatrixXd information_;  // 约束边的信息矩阵（置信度）
    Eigen::MatrixXd robust_information_; // 柯西代价函数的信息矩阵
};

// 曲线拟合边约束
class CurveFittingFactor : public Factor
{
public:
    CurveFittingFactor(Eigen::VectorXd means)
    {
        means_ = means;
    }

    void computerError() override
    {
        Eigen::MatrixXd new_information = information_;
        double a_op = vertices_[0]->data_(0);
        double b_op = vertices_[0]->data_(1);
        double c_op = vertices_[0]->data_(2);
        error_(0) = std::exp(a_op * means_(0) * means_(0) + b_op * means_(0) + c_op)-means_(1);
        if(loss_function_)
        {
            double u_x = std::sqrt( error_.transpose() * information_ * error_ );
            loss_function_->computerJacobian(u_x);
            robust_information_ = 1.0 / (u_x+0.000001) * loss_function_->jacobians_ * information_;
            new_information = robust_information_;
        }
        squared_error_ = 0.5 * error_.transpose()*new_information*error_;  // 使用新的信息矩阵构建平方误差项
    }

    void computerJacobian() override
    {
        Eigen::MatrixXd jacobians_abc(dim_resiual_, vertices_[0]->dimension_);
        double a_op = vertices_[0]->data_(0);
        double b_op = vertices_[0]->data_(1);
        double c_op = vertices_[0]->data_(2);
        double exp_x = std::exp(a_op*means_(0)*means_(0) + b_op*means_(0) + c_op);
        jacobians_abc << exp_x*means_(0)*means_(0), exp_x*means_(0), exp_x;  // 残差对顶点（优化变量）求导
        jacobians_[0] = jacobians_abc;
    }

};

class LSSolver
{
public:
    LSSolver() : currentChi_(0.0)
    {}



    bool solve(int iterations)
    {
        if(factors_.size() == 0 || vertices_.size() == 0)
        {
            std::cerr<<"\n cannot solver problem without edges or vertices_"<<std::endl;
            return false;
        }

        // 设置顶点
        setOrdering();
        // 构建hessian矩阵
        makeHessian();
        int iter = 0;
        while(iter < iterations)
        {
            //计算当前总的残差
            currentChi_ = 0.0;
            for(auto factor : factors_)
            {
                currentChi_ += factor->squared_error_;
            }
            // std::cout.precision(20);
            std::cout<<"iter: "<<iter<<" , "<<"currentChi_: "<<currentChi_<<std::endl;
            solverLinearSystem();  // 求解正规矩阵 ldlt求解
            if(delta_x_.squaredNorm() <= 1e-6)  // squaredNorm针对向量表示L2距离
                break;
            updateStates();  //更新状态
            makeHessian();
            iter++;
        }

        return true;
    }

    void solverLinearSystem()
    {
        delta_x_ = Hessian_.ldlt().solve(b_);
    }

    void updateStates()
    {
        // 更新状态量, factor
        for(auto &vertex : vertices_)
        {
            ulong idx = vertex->ordering_id_;
            ulong dim = vertex->dimension_;
            Eigen::VectorXd delta = delta_x_.segment(idx, dim);
            vertex->data_ += delta;
        }
	}

	
	void makeHessian()
    {
        Eigen::MatrixXd H(all_states_size, all_states_size);
        Eigen::VectorXd b(all_states_size);
        H.setZero();
        b.setZero();

        for(auto& factor : factors_)    // 一元边，待优化变量单一
        {
            factor->computerError();    // 计算约束边的误差
            factor->computerJacobian(); // 计算约束边的雅克比矩阵

            // 
            auto jacobians = factor->jacobians_;
            auto vertices = factor->vertices_;
            assert(jacobians.size() == vertices.size());  // 断言，条件为错误，将中断
            for(size_t i = 0; i<vertices.size(); ++i)
            {
                // cout<<"vertices i is = "<< i <<endl;
                auto v_i = vertices[i];
                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id_;  //用于计算hessian矩阵中的位置
                ulong dim_i = v_i->dimension_;      //顶点维度
                Eigen::MatrixXd JtW; // 雅克比矩阵的转置乘以信息矩阵
                if(factor->loss_function_)
                    JtW = jacobian_i.transpose() * factor->robust_information_;
                else
                    JtW = jacobian_i.transpose() * factor->information_;
                for(size_t j = i; j<vertices.size(); ++j)  // j位置处的雅克比矩阵
                {
                    auto v_j = vertices[j];
                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id_;
                    ulong dim_j = v_j->dimension_;

                    // 计算hessian矩阵 J^T * W *J = Hessian
                    Eigen::MatrixXd hessian = JtW * jacobian_j;

                    // 构建hessian矩阵，按照i，j位置
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;  //noalias标记没有混淆，行雅克比
                    if(j != i)
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();  // 列雅克比
                }
                b.segment(index_i, dim_i).noalias() -= JtW * factor->error_;  //segment（i，j）表示从标号i开始的j个元素
            }
        }

        // 给hessian和b矩阵赋值
        Hessian_ = H;
        b_ = b;

        delta_x_ = Eigen::VectorXd::Zero(all_states_size);
    }

    void setOrdering()
    {
        ulong ordering_id = 0;
        for(auto vertex : vertices_)
        {
            vertex->ordering_id_ = ordering_id;
            ordering_id += vertex->dimension_;   // 顶点在矩阵中的位置（将优化变量维度计算上）
        }
        all_states_size = ordering_id;  // 待优化状态变量大小
    }
	
    void addFactors(std::shared_ptr<Factor> factor, std::shared_ptr<LossFunction> loss_function, std::vector<std::shared_ptr<Vertex>> vertex, int dim)
    {
        factors_.push_back(factor);
        factor->setParametersDims(dim, vertex);
        factor->setLossFunction(loss_function);
    }
	    // 求解器中添加顶点
    void addVertex(std::shared_ptr<Vertex> vertex)
    {
        vertices_.push_back(vertex);
    }


    std::vector<std::shared_ptr<Vertex>> vertices_;  // 待估计参数
    std::vector<std::shared_ptr<Factor>> factors_;   // 约束边

    Eigen::MatrixXd Hessian_;
    Eigen::MatrixXd b_;
    Eigen::VectorXd delta_x_;

    int all_states_size;
    double currentChi_;


};



int main(int agrc, char** argv)
{

    // // 1.创建A矩阵 6行5列，随机数【-1，1】之间
    // // Eigen::MatrixXd matrA = EIgen::MatrixXd::Random(6,5);
    // Eigen::MatrixXd matrA(6,5);
    // matrA = Eigen::MatrixXd::Random(6,5);
    // matrA *= 10;  // 随机数扩大十倍

    // cout<<"matrA = "<<matrA<<endl;     
    // // 2.SVD矩阵分解求解
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // // cout<<"SVD singual value is : "<<svd.singularValues()<<endl;
    // // cout<<"SVD ComputeFullV is : "<<svd.matrixV()<<endl;
    // // cout<<"SVD ComputeFullU is : "<<svd.matrixU()<<endl;
    // // cout<<"SVD ComputeFullV min singal value is : "<<svd.matrixV().rightCols(1)<<endl;

    // // 3.AX=b
    // Eigen::VectorXd matrXT = Eigen::VectorXd::Random(5,1);
    // Eigen::VectorXd matrB = matrA * matrXT;

    // cout<<"matrXT = "<<matrXT.transpose()<<endl;

    // // 4.QR分解求解AX=b
    // Eigen::VectorXd matrX = matrA.colPivHouseholderQr().solve(matrB);
    // cout<<"matrX = "<<matrX.transpose()<<endl;

    // 5.BA优化，使用鲁邦核函数
    double a=0.5, b=0.6, c=2.0;
    int N = 100;
    double w_sigma = 0.05;

    // 5.1 添加噪声
    std::default_random_engine generator; //噪声生成器
    std::normal_distribution<double> noise(0., w_sigma);

    // 5.2 定义求解器
    LSSolver solver;

    // 5.3 定义待优化变量（顶点）
    std::shared_ptr<Vertex> abc(new Vertex(3));
    abc->data_ << 0., 0., 0.; // 初始顶点数据
    solver.addVertex(abc);

    for(int i=0; i<N; ++i)
    {
        double x = i/100.0;
        double n = noise(generator);
        double y = std::exp(a*x*x + b*x + c) + n;
        Eigen::VectorXd means(2);
        means << x, y;

        // 添加约束边
        std::shared_ptr<CurveFittingFactor> factor(new CurveFittingFactor(means));
        // 信息矩阵(协方差矩阵的倒数，表示置信度)
        Eigen::MatrixXd information_matrix(1, 1);
        information_matrix << 1./(w_sigma * w_sigma);

        factor->setInformationMatrix(information_matrix);  // 给约束边设置信息矩阵

        // 求解器添加约束边, 鲁邦核函数，顶点优化变量, 误差维度
        solver.addFactors(factor, std::shared_ptr<LossFunction>(new CauchyLoss(0.5)), std::vector<std::shared_ptr<Vertex>>{abc}, 1);

    }

    solver.solve(50);

    std::cout<<"BA optimization after parameter is abc = "<<abc->data_.transpose()<<std::endl;

    return 0;
}