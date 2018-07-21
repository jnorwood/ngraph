/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/runtime/gpu/emitter.hpp"
#include <type_traits>

namespace ngraph
{
    namespace op
    {
        namespace gpu
        {
            class MemoryWrappedNode_Base : public Node
            {
            public:
                MemoryWrappedNode_Base(const std::shared_ptr<Node>& node)
                    : Node(node->description(), node->get_arguments()) {}
                virtual ~MemoryWrappedNode_Base() = default;
                virtual void emit(runtime::gpu::GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const std::vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                  const std::vector<runtime::gpu::GPU_TensorViewWrapper>& out) {}
            };

            template <typename NODE_TYPE>
            class MemoryWrappedNode : public MemoryWrappedNode_Base
            {
            public:
                MemoryWrappedNode(const std::shared_ptr<NODE_TYPE>& node)
                    : MemoryWrappedNode_Base(node)
                    , m_node(node)
                    , m_emitter(this)
                {
                    // add node's outputs to wrapped node
                    size_t i = 0;
                    for (auto& output : node->get_outputs())
                    {
                        this->m_outputs.emplace_back(this, i, output.get_tensor_view());
                    }

                    // add constant memory input
                    i = this->m_inputs.size();
                    for (auto& data : m_emitter.get_constants())
                    {
                        auto constant = std::make_shared<op::Constant>(
                            ngraph::element::from<typename std::remove_reference<decltype(data[0])>::type>(),
                            Shape{data.size()}, data);
                        this->m_inputs.emplace_back(
                            this, i++, constant->get_outputs().at(0));
                    }

                    // add worskapce output
                    for (auto& workspace : m_emitter.get_workspaces())
                    {
                         this->add_output(m_node->get_element_type(), workspace);
                    }
                }

                void emit(runtime::gpu::GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const std::vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                          const std::vector<runtime::gpu::GPU_TensorViewWrapper>& out) override
                    {
                        m_emitter.emit(external_function, writer, args, out);
                    }

                std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
                {
                    auto new_node = std::dynamic_pointer_cast<NODE_TYPE>(m_node->copy_with_new_args(new_args));
                    return std::make_shared<MemoryWrappedNode<NODE_TYPE>>(new_node);
                }

                const std::shared_ptr<NODE_TYPE> get() const { return m_node; }

                private:
                std::shared_ptr<NODE_TYPE> m_node;
                runtime::gpu::Emitter<NODE_TYPE> m_emitter;
            };



        }
    }
}
