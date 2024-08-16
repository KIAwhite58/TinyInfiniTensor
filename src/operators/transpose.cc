#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        int rank = A->getRank();

        // =================================== 作业 ===================================
       if (transposePermute.size() != rank) {
            return std::nullopt; // 当 transposePermute 大小不匹配时，返回 nullopt
        }

        // 根据 permute 向量重排输出维度
        for (int i = 0; i < rank; ++i)
        {
            if (transposePermute[i] < 0 || transposePermute[i] >= rank)
            {
                return std::nullopt; // 防止超出范围的索引
            }
            output_dim[i] = input_dim[transposePermute[i]];
        }
        // =================================== 作业 ===================================

        return std::nullopt;
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
