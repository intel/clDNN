/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "jitter.h"
#include "tensor_type.h"

namespace {
    class JitTerm
    {
    public:
        JitTerm(std::string text)
            : text(std::move(text))
        {}

        std::string str() const { return text; }

        JitTerm gt(const JitTerm& rhs) const
        {
            return {"(" + text + ">" + rhs.str() + ")"};
        }

        JitTerm ge(const JitTerm& rhs) const
        {
            return {"(" + text + ">=" + rhs.str() + ")"};
        }

        JitTerm le(const JitTerm& rhs) const
        {
            return {"(" + text + "<=" + rhs.str() + ")"};
        }

        JitTerm eq(const JitTerm& rhs) const
        {
            return {"(" + text + "==" + rhs.str() + ")"};
        }

    private:
        std::string text;
    };

    JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs)
    {
        return {"(" + lhs.str() + " + " + rhs.str() + ")"};
    }

    JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs)
    {
        return {"(" + lhs.str() + " - " + rhs.str() + ")"};
    }

    JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs)
    {
        return {"(" + lhs.str() + " * " + rhs.str() + ")"};
    }

    JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs)
    {
        return {"(" + lhs.str() + " / " + rhs.str() + ")"};
    }

    JitTerm neg(const JitTerm& arg)
    {
        return {"(-" + arg.str() + ")"};
    }

    JitTerm ternary(const JitTerm& condition,
                    const JitTerm& true_expr,
                    const JitTerm& false_expr)
    {
        return {"(" + condition.str() + " ? " + true_expr.str() + " : "
                + false_expr.str() + ")"};
    }
    JitTerm isinf(const JitTerm& arg)
    {
        return {"(isinf(" + arg.str() + "))"};
    }

    JitTerm exp(const JitTerm& arg)
    {
        return {"(exp(" + arg.str() + "))"};
    }

    JitTerm log(const JitTerm& arg)
    {
        return {"(log(" + arg.str() + "))"};
    }

    JitTerm operator"" _jit(const char* str, size_t)
    {
        return {str};
    }
} // namespace

namespace kernel_selector {

    std::string toCLType(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::INT8: return GetTypeName<int8_t>();
        case WeightsType::UINT8: return GetTypeName<uint8_t>();
        case WeightsType::F16:  return "half";
        case WeightsType::F32:  return GetTypeName<float>();
        default: return "";
        }
    }

    std::string toCLType(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::INT8:    return GetTypeName<int8_t>();
        case Datatype::UINT8:   return GetTypeName<uint8_t>();
        case Datatype::INT16:   return GetTypeName<int16_t>();
        case Datatype::UINT16:  return GetTypeName<uint16_t>();
        case Datatype::INT32:   return GetTypeName<int32_t>();
        case Datatype::UINT32:  return GetTypeName<uint32_t>();
        case Datatype::INT64:   return GetTypeName<int64_t>();
        case Datatype::F16:     return "half";
        case Datatype::F32:     return GetTypeName<float>();
        default: return "";
        }
    }

    std::string getMeanOpString(MeanOp op)
    {
        switch (op)
        {
        case MeanOp::NONE:   return "val";
        case MeanOp::DIV:    return "val/mean_val";
        case MeanOp::MUL:    return "val*mean_val";
        case MeanOp::SUB:    return "val-mean_val";
        default: return "";
        }
    }

    std::string toCodeString(float val) {
        if (std::isinf(val))
            return std::signbit(val) ? "-INFINITY" : "INFINITY";
        std::stringstream ss;
        // Workaround GCC compiler/STL bug
        ss << "as_float(0x" << std::hex << *reinterpret_cast<uint32_t*>(&val) << ")";

        ss << " /*" << std::scientific << val << "*/";
        return ss.str();
    }

    std::string toCodeString(double val) {
        if (std::isinf(val))
            return std::signbit(val) ? "-INFINITY" : "INFINITY";
        std::stringstream ss;
        // Workaround GCC compiler/STL bug
        ss << "as_double(0x" << std::hex << *reinterpret_cast<uint64_t*>(&val) << ")";

        ss << " /*" << std::scientific << val << "*/";
        return ss.str();
    }

    JitDefinitions JitConstants::GetDefinitions() const
    {
        JitDefinitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->GetDefinitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TensorBaseTJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename DType, typename Layout>
    class TensorBaseTJitConstant : public JitConstant
    {
    protected:
        TensorBaseTJitConstant(const std::string& name) : JitConstant(name) {}

    public:

        JitDefinitions GetDefinitions(const Tensor::TensorBaseT<DType, Layout>& t) const
        {
            JitDefinitions definitions{
            { _name + "_OFFSET",        toCodeString(t.GetFirstElementOffset()) },
            { _name + "_VIEW_OFFSET",   toCodeString(t.GetViewOffset()) },
            { _name + "_LENGTH",        toCodeString(t.LogicalSize()) },
            { _name + "_DIMS",          toCodeString(t.GetDims().size()) },
            { _name + "_SIMPLE",        toCodeString(t.SimpleLayout()) },
            { _name + "_LAYOUT_" + toString(t.GetLayout()), "1" },
            };

            auto type_defs = MakeTypeJitConstants(t.GetDType(), _name).GetDefinitions();
            definitions.insert(definitions.end(), type_defs.begin(), type_defs.end());


            definitions.push_back({ _name + "_SIZE",        toCodeString(t.GetDims().size()) });
            definitions.push_back({ _name + "_SIZES",       toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; }) });
            definitions.push_back({ _name + "_PITCHES",     toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; }) });
            definitions.push_back({ _name + "_PAD_BEFORE",  toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.before; }) });
            definitions.push_back({ _name + "_PAD_AFTER",   toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.after; }) });

            return definitions;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DataTensorJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class DataTensorJitConstant : public TensorBaseTJitConstant<Datatype, DataLayout>
    {
        const DataTensor _tensor;

    public:
        DataTensorJitConstant(const std::string& name, const DataTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

        JitDefinitions GetDefinitions() const override;
    };

    JitDefinitions DataTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
        { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
        { _name + "_SIZE_Z",        toCodeString(_tensor.Z().v) },
        { _name + "_FEATURE_NUM",   toCodeString(_tensor.Feature().v) },
        { _name + "_ROI_NUM",       toCodeString(_tensor.ROI().v) },
        { _name + "_BATCH_NUM",     toCodeString(_tensor.Batch().v) },
        { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
        { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
        { _name + "_Z_PITCH",       toCodeString(_tensor.Z().pitch) },
        { _name + "_FEATURE_PITCH", toCodeString(_tensor.Feature().pitch) },
        { _name + "_ROI_PITCH",     toCodeString(_tensor.ROI().pitch) },
        { _name + "_BATCH_PITCH",   toCodeString(_tensor.Batch().pitch) },
        { _name + "_PAD_BEFORE_SIZE_X",        toCodeString(_tensor.X().pad.before) },
        { _name + "_PAD_BEFORE_SIZE_Y",        toCodeString(_tensor.Y().pad.before) },
        { _name + "_PAD_BEFORE_SIZE_Z",        toCodeString(_tensor.Z().pad.before) },
        { _name + "_PAD_BEFORE_FEATURE_NUM",   toCodeString(_tensor.Feature().pad.before) },
        { _name + "_PAD_BEFORE_BATCH_NUM",     toCodeString(_tensor.Batch().pad.before) },
        { _name + "_PAD_AFTER_SIZE_X",         toCodeString(_tensor.X().pad.after) },
        { _name + "_PAD_AFTER_SIZE_Y",         toCodeString(_tensor.Y().pad.after) },
        { _name + "_PAD_AFTER_SIZE_Z",         toCodeString(_tensor.Z().pad.after) },
        { _name + "_PAD_AFTER_FEATURE_NUM",    toCodeString(_tensor.Feature().pad.after) },
        { _name + "_PAD_AFTER_BATCH_NUM",      toCodeString(_tensor.Batch().pad.after) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DataTensor& value)
    {
        return std::static_pointer_cast<JitConstant>(std::make_shared<DataTensorJitConstant>(name, value));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightTensorJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class WeightTensorJitConstant : public TensorBaseTJitConstant<WeightsType, WeightsLayout>
    {
        const WeightsTensor _tensor;

    public:
        WeightTensorJitConstant(const std::string& name, const WeightsTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

        JitDefinitions GetDefinitions() const override;
    };

    JitDefinitions WeightTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
            { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
            { _name + "_SIZE_Z",        toCodeString(_tensor.Z().v) },
            { _name + "_IFM_NUM",       toCodeString(_tensor.IFM().v) },
            { _name + "_OFM_NUM",       toCodeString(_tensor.OFM().v) },
            { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
            { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
            { _name + "_Z_PITCH",       toCodeString(_tensor.Z().pitch) },
            { _name + "_IFM_PITCH",     toCodeString(_tensor.IFM().pitch) },
            { _name + "_OFM_PITCH",     toCodeString(_tensor.OFM().pitch) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const WeightsTensor& value)
    {
        return std::static_pointer_cast<JitConstant>(std::make_shared<WeightTensorJitConstant>(name, value));
    }

    std::shared_ptr<JitConstant>
    MakeActivationJitConstants(ActivationFunction activation_function,
                               const std::string& suffix,
                               bool use_type_parameter)
    {
        std::string name = "ACTIVATION" + suffix;

        // See the comment in the jitter.h regarding `use_type_parameter`.
        // The "CAT" macro is expected to be defined through the inlcusion of
        // 'common.cl' in the kernel.
        auto type_handler =
            [use_type_parameter](const std::string& prefix,
                                 const std::string& suffix) -> std::string {
            if (!use_type_parameter)
                return prefix + "UNIT" + suffix;

            std::string result = "jit_type";

            // Process the prefix first, otherwise when doing "CAT(TO_,
            // CAT(NAME, _TYPE))" the second concatenation will be expanded
            // fully first resulting in something like "TO_float".
            if (!prefix.empty())
                result = "CAT(" + prefix + ", " + result + ")";

            if (!suffix.empty())
                result = "CAT(" + result + ", " + suffix + ")";

            return result;
        };

        const JitTerm one{type_handler("", "_VAL_ONE")};
        const JitTerm zero{type_handler("", "_VAL_ZERO")};
        const JitTerm input{"input"};
        auto max_func = [type_handler](const JitTerm& lhs,
                                       const JitTerm& rhs) -> JitTerm {
            return {"(" + type_handler("", "_MAX_FUNC") + "(" + lhs.str() + ", "
                    + rhs.str() + "))"};
        };
        auto min_func = [type_handler](const JitTerm& lhs,
                                       const JitTerm& rhs) -> JitTerm {
            return {"(" + type_handler("", "_MIN_FUNC") + "(" + lhs.str() + ", "
                    + rhs.str() + "))"};
        };
        auto to_type = [type_handler](const JitTerm& arg) -> JitTerm {
            return {type_handler("TO_", "_TYPE") + "(" + arg.str() + ")"};
        };

        std::string macro_def = name
            + (use_type_parameter ? "(jit_type, input, m, n)" : "(input, m, n)");
        std::string macro_def_grad = name
            + (use_type_parameter ? "(jit_type, input_grad, input, m, n)"
                                  : "(input_grad, input, m, n)");
        // TODO: use native_exp and use cast for APL
        switch (activation_function)
        {
        case ActivationFunction::LOGISTIC:
            return MakeJitConstant(macro_def,
                                   (one / (one + exp(neg(input)))).str());
        case ActivationFunction::HYPERBOLIC_TAN:
            return MakeJitConstant(macro_def, "(tanh(input))");
        case ActivationFunction::RELU:
            return MakeJitConstant(macro_def, max_func(zero, input).str());
        case ActivationFunction::RELU_NEGATIVE_SLOPE:
        {
            const JitTerm slope = to_type("m"_jit);
            return MakeJitConstant(
                macro_def,
                ternary(isinf(slope),
                        ternary(input.ge(zero), input, neg(slope)),
                        max_func(input, zero) + (slope * min_func(input, zero)))
                    .str());
        }
        case ActivationFunction::ELU:
        {
            auto alpha = "m"_jit;
            return MakeJitConstant(
                macro_def,
                (max_func(input, zero)
                 + (to_type(alpha) * (exp(min_func(input, zero)) - one)))
                    .str());
        }
        case ActivationFunction::CLAMP:
            return MakeJitConstant(
                macro_def,
                max_func(to_type("m"_jit), min_func("n"_jit, input)).str());
        case ActivationFunction::SOFTRELU:
            return MakeJitConstant(macro_def, log(one + exp(input)).str());
        case ActivationFunction::ABS:
            return MakeJitConstant(macro_def, "(fabs(input))");
        case ActivationFunction::LINEAR:
            return MakeJitConstant(macro_def, "(m*input + n)");
        case ActivationFunction::SQUARE:
            return MakeJitConstant(macro_def, "(input*input)");
        case ActivationFunction::SQRT:
            return MakeJitConstant(macro_def, "(sqrt(input))");
        case ActivationFunction::SIN:
            return MakeJitConstant(macro_def, "(sin(input))");
        case ActivationFunction::ASIN:
            return MakeJitConstant(macro_def, "(asin(input))");
        case ActivationFunction::SINH:
            return MakeJitConstant(macro_def, "(sinh(input))");
        case ActivationFunction::COS:
            return MakeJitConstant(macro_def, "(cos(input))");
        case ActivationFunction::ACOS:
            return MakeJitConstant(macro_def, "(acos(input))");
        case ActivationFunction::COSH:
            return MakeJitConstant(macro_def, "(cosh(input))");
        case ActivationFunction::LOG:
            return MakeJitConstant(macro_def, "(log(input))");
        case ActivationFunction::LOG2:
            return MakeJitConstant(macro_def, "(log2(input))");
        case ActivationFunction::EXP:
            return MakeJitConstant(macro_def, "(exp(input))");
        case ActivationFunction::RELU_GRAD:
            return MakeJitConstant(
                macro_def_grad,
                ("input_grad"_jit * ternary(input.gt(zero), one, zero)).str());
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD:
        {
            auto slope = "m"_jit;
            return MakeJitConstant(
                macro_def_grad,
                ("input_grad"_jit
                 * (ternary(input.gt(zero), one, zero)
                    + (to_type(slope) * ternary(input.le(zero), one, zero))))
                    .str());
        }
        case ActivationFunction::NONE_GRAD:
            return MakeJitConstant(macro_def_grad, "input_grad");
        case ActivationFunction::TAN:
            return MakeJitConstant(macro_def, "(tan(input))");
        case ActivationFunction::ATAN:
            return MakeJitConstant(macro_def, "(atan(input))");
        case ActivationFunction::FLOOR:
            return MakeJitConstant(macro_def, "(floor(input))");
        case ActivationFunction::CEIL:
            return MakeJitConstant(macro_def, "(ceil(input))");
        case ActivationFunction::NEGATIVE:
            return MakeJitConstant(macro_def, "(-input)");
        case ActivationFunction::NOT:
            return MakeJitConstant(
                macro_def,
                ternary(input.eq(zero), one, zero)
                    .str()); // the workaround for OpenCL's vector type result (!input)
        case ActivationFunction::NONE:
        default:
            return MakeJitConstant(macro_def, "input");
        }
    }

    JitConstants MakeTypeJitConstants(Datatype dataType, const std::string& macroName)
    {
        std::string type;
        std::string max_val;
        std::string min_val;
        std::string val_one;
        std::string val_zero;
        std::string to_type;
        std::string to_type_sat;
        std::string max_func;
        std::string min_func;
        std::string type_size;
        bool is_fp;
        switch (dataType)
        {
        case Datatype::INT8:
            type = "char";
            max_val = "CHAR_MAX";
            min_val = "CHAR_MIN";
            val_one = "(char) 1";
            val_zero = "(char) 0";
            to_type = "convert_char(v)";
            to_type_sat = "convert_char_sat(v)";
            max_func = "max";
            min_func = "min";
            type_size = "1";
            is_fp = false;
            break;
        case Datatype::UINT8:
            type = "uchar";
            max_val = "UCHAR_MAX";
            min_val = "0";
            val_one = "(uchar) 1";
            val_zero = "(uchar) 0";
            to_type = "convert_uchar(v)";
            to_type_sat = "convert_uchar_sat(v)";
            max_func = "max";
            min_func = "min";
            type_size = "1";
            is_fp = false;
            break;
        case Datatype::INT32:
            type = "int";
            max_val = "INT_MAX";
            min_val = "INT_MIN";
            val_one = "(int) 1";
            val_zero = "(int) 0";
            to_type = "convert_int(v)";
            to_type_sat = "convert_int_sat(v)";
            max_func = "max";
            min_func = "min";
            type_size = "4";
            is_fp = false;
            break;
        case Datatype::UINT32:
            type = "uint";
            max_val = "UINT_MAX";
            min_val = "0";
            val_one = "(uint) 1";
            val_zero = "(uint) 0";
            to_type = "convert_uint(v)";
            to_type_sat = "convert_uint_sat(v)";
            max_func = "max";
            min_func = "min";
            type_size = "4";
            is_fp = false;
            break;
        case Datatype::INT64:
            type = "long";
            max_val = "LONG_MAX";
            min_val = "LONG_MIN";
            val_one = "(long) 1";
            val_zero = "(long) 0";
            to_type = "convert_long(v)";
            to_type_sat = "convert_long_sat(v)";
            max_func = "max";
            min_func = "min";
            type_size = "8";
            is_fp = false;
            break;
        case Datatype::F16:
            type = "half";
            max_val = "HALF_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0h";
            val_zero = "0.0h";
            to_type = "convert_half(v)";
            to_type_sat = "convert_half(v)";
            max_func = "fmax";
            min_func = "fmin";
            type_size = "2";
            is_fp = true;
            break;
        default:
            type = "float";
            max_val = "FLT_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0f";
            val_zero = "0.0f";
            to_type = "convert_float(v)";
            to_type_sat = "convert_float(v)";
            max_func = "fmax";
            min_func = "fmin";
            type_size = "4";
            is_fp = true;
            break;
        }

        return JitConstants
        {
            MakeJitConstant(macroName+"_TYPE",              type),
            MakeJitConstant(macroName+"_VAL_MAX",           max_val),
            MakeJitConstant(macroName+"_VAL_MIN",           min_val),
            MakeJitConstant(macroName+"_VAL_ONE",           val_one),
            MakeJitConstant(macroName+"_VAL_ZERO",          val_zero),
            MakeJitConstant("TO_"+macroName+"_TYPE(v)",     to_type),
            MakeJitConstant("TO_"+macroName+"_TYPE_SAT(v)", to_type_sat),
            MakeJitConstant(macroName+"_MAX_FUNC",          max_func),
            MakeJitConstant(macroName+"_MIN_FUNC",          min_func),
            MakeJitConstant(macroName + "_TYPE_SIZE",       type_size),
            MakeJitConstant(macroName+"_IS_FP",             is_fp),
        };
    }
    JitConstants MakeTypeJitConstants(WeightsType weightsType, const std::string& macroName)
    {
      switch (weightsType)
        {
        case WeightsType::UNSUPPORTED:
            return MakeTypeJitConstants(Datatype::UNSUPPORTED, macroName);
        case WeightsType::F16:
            return MakeTypeJitConstants(Datatype::F16, macroName);
        case WeightsType::F32:
            return MakeTypeJitConstants(Datatype::F32, macroName);
        case WeightsType::INT8:
            return MakeTypeJitConstants(Datatype::INT8, macroName);
        case WeightsType::UINT8:
            return MakeTypeJitConstants(Datatype::UINT8, macroName);
        }
        assert(false || "Unreachable!");
        // FIXME: Is there some builtin_unreachable available?
        return MakeTypeJitConstants(Datatype::UNSUPPORTED, macroName);
    }

    JitConstants MakeActivationJitConstants(const base_activation_params& params,
                                            const std::string& suffix,
                                            bool use_type_parameter)
    {
        return JitConstants{MakeJitConstant("NL_M" + suffix, params.m),
                            MakeJitConstant("NL_N" + suffix, params.n),
                            MakeActivationJitConstants(
                                params.function, suffix, use_type_parameter)};
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeLoopUnrollParamsJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount)
    {
        JitConstants jit{
            MakeJitConstant("LOOP0(VAR, STMT)", ""),
            MakeJitConstant("LOOP1(VAR, STMT)", "(STMT); (VAR)++;"),
        };

        for (uint32_t i = 2; i <= loopCount + 1; i++)
        {
            jit.AddConstant({
                MakeJitConstant("LOOP" + toCodeString(i) + "(VAR, STMT)", "LOOP" + toCodeString(i - 1) + "(VAR, STMT); (STMT); (VAR)++;"),
                });
        }

        jit.AddConstant({
            MakeJitConstant("LOOP(N, VAR, STMT)", "CAT(LOOP, N)((VAR), (STMT))"),
            });

        return jit;
    }

}
