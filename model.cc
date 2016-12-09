// Copyright (C) 2016 West Virginia University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of West Virginia University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@mail.wvu.edu)

#include "model.h"
#include <iostream>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "shader_program.h"
#include "transformations.h"

namespace wvu {
Model::Model(const Eigen::Vector3f& orientation,
             const Eigen::Vector3f& position,
             const Eigen::MatrixXf& vertices) {
  orientation_ = orientation;
  position_ = position;
  vertices_ = vertices;
  vertex_buffer_object_id_ = 0;
  vertex_array_object_id_ = 0;
  element_buffer_object_id_ = 0;
  speed_rot = 1;
  speed_bob = 1;
  rot_speed_ = 0.0f;
  bob_speed_ = 0.0f;
}

Model::Model(const Eigen::Vector3f& orientation,
             const Eigen::Vector3f& position,
             const Eigen::MatrixXf& vertices,
             const std::vector<GLuint>& indices) {
  orientation_ = orientation;
  position_ = position;
  vertices_ = vertices;
  indices_ = indices;
  vertex_buffer_object_id_ = 0;
  vertex_array_object_id_ = 0;
  element_buffer_object_id_ = 0;
  speed_rot = 1;
  speed_bob = 1;
  rot_speed_ = 0.0f;
  bob_speed_ = 0.0f;
}

Model::~Model() {
    	glDeleteVertexArrays(1, &vertex_array_object_id_);
  	glDeleteBuffers(1, &vertex_buffer_object_id_);
}

// Builds the model matrix from the orientation and position members.
Eigen::Matrix4f Model::ComputeModelMatrix() 
{
	Eigen::Vector3f bob = Eigen::Vector3f(0.0f, sin(speed_bob)*0.05f, 0.0f); 
	Eigen::Matrix4f translation = ComputeTranslationMatrix(position_ + bob);
	Eigen::Matrix4f rotation = ComputeRotationMatrix(orientation_.normalized(), orientation_.norm() * speed_rot);
	Eigen::Matrix4f scale = ComputeScalingMatrix(1);

  	return translation * rotation * scale;
}

void Model::set_rotation_speed(const float rotation_speed) {
	rot_speed_ = rotation_speed;
}

void Model::set_bob_speed(const float bob_speed) {
	bob_speed_ = bob_speed;
}

// Setters set members by *copying* input parameters.
void Model::set_orientation(const Eigen::Vector3f& orientation) {
  orientation_ = orientation;
}

// Setters set members by *copying* input parameters.
void Model::set_position(const Eigen::Vector3f& position) {
  position_ = position;
}

float Model::rotation_speed() {
	return rot_speed_;
}

float Model::bob_speed() {
	return bob_speed_;
}

Eigen::Vector3f* Model::mutable_orientation() {
  return &orientation_;
}

Eigen::Vector3f* Model::mutable_position() {
  return &position_;
}

const Eigen::Vector3f& Model::orientation() {
  return orientation_;
}

const Eigen::Vector3f& Model::position() {
  return position_;
}

const Eigen::MatrixXf& Model::vertices() const {
  return vertices_;
}

const std::vector<GLuint>& Model::indices() const {
  return indices_;
}

const GLuint Model::vertex_buffer_object_id() const {
  return vertex_buffer_object_id_;
}

const GLuint Model::vertex_buffer_object_id() {
  return vertex_buffer_object_id_;
}

const GLuint Model::vertex_array_object_id() const {
  return vertex_array_object_id_;
}

const GLuint Model::vertex_array_object_id() {
  return vertex_array_object_id_;
}

const GLuint Model::element_buffer_object_id() const {
  return element_buffer_object_id_;
}

const GLuint Model::element_buffer_object_id() {
  return element_buffer_object_id_;
}

void Model::SetVerticesIntoGpu() 
{
	//Create and Set the VAO
  	constexpr int kNumVertexArrays = 1;
  	glGenVertexArrays(kNumVertexArrays, &vertex_array_object_id_);
  	glBindVertexArray(vertex_array_object_id_);

  	//Create and Set the VBO
  	GLuint vertex_buffer_object_id;
  	glGenBuffers(1, &vertex_buffer_object_id);
  	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object_id);
  	const Eigen::MatrixXf& vertices = vertices_;
  	const int vertices_size_in_bytes =
      		vertices.rows() * vertices.cols() * sizeof(vertices(0, 0));
  	glBufferData(GL_ARRAY_BUFFER,
               	     vertices_size_in_bytes,
                     vertices.data(),
                     GL_STATIC_DRAW);
  	constexpr GLuint kIndex = 0;
  	constexpr GLuint kNumElementsPerVertex = 3;
  	constexpr GLuint kStride = kNumElementsPerVertex * sizeof(vertices(0, 0));
  	const GLvoid* offset_ptr = nullptr;
	
	// configure verts
  	glVertexAttribPointer(kIndex, kNumElementsPerVertex, GL_FLOAT, GL_FALSE,
                              kStride, offset_ptr);
  	glEnableVertexAttribArray(kIndex);

	// Configure the texels.
  	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                        kStride, offset_ptr);
  	glEnableVertexAttribArray(1);

  	glBindBuffer(GL_ARRAY_BUFFER, 0);
  	vertex_buffer_object_id_ = vertex_buffer_object_id;

	//Create and Set the EBO
  	GLuint element_buffer_object_id;
  	glGenBuffers(1, &element_buffer_object_id);
  	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object_id);
  	const std::vector<GLuint>& indices = indices_;
  	const int indices_size_in_bytes = indices.size() * sizeof(indices[0]);
  	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               	     indices_size_in_bytes,
                     indices.data(),
                     GL_STATIC_DRAW);
  	element_buffer_object_id_ = element_buffer_object_id;

  	// Disable the created VAO.
  	glBindVertexArray(0);
}

void Model::Draw(const ShaderProgram& shader_program,
                 const Eigen::Matrix4f& projection,
                 const Eigen::Matrix4f& view,
		 const GLuint texture_id) 
{
  	// Get the locations of the uniform variables.
  	const GLint model_location = glGetUniformLocation(shader_program.shader_program_id(), "model");
  	const GLint view_location = glGetUniformLocation(shader_program.shader_program_id(), "view");
  	const GLint projection_location = glGetUniformLocation(shader_program.shader_program_id(), "projection");
	const GLint vertex_color_location = glGetUniformLocation(shader_program.shader_program_id(), "vertex_color");

  	// The model transformation must be computed using ComputeModelMatrix().
	speed_rot = rotation_speed() * static_cast<GLfloat>(glfwGetTime());
	speed_bob = bob_speed() * static_cast<GLfloat>(glfwGetTime());
  	const Eigen::Matrix4f model = ComputeModelMatrix();

	// Bind texture.
  	glBindTexture(GL_TEXTURE_2D, texture_id);

	glUniformMatrix4fv(model_location, 1, GL_FALSE, model.data());
  	glUniformMatrix4fv(view_location, 1, GL_FALSE, view.data());
  	glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection.data());
	GLfloat color_scalar = static_cast<GLfloat>(glfwGetTime());

	//send the color
	Eigen::Vector4f color(0.5f, 0.5f, 0.5f, 1.0f);
  	glUniform4fv(vertex_color_location, 1, color.data());
	
	glBindVertexArray(vertex_array_object_id_);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  	glDrawElements(GL_TRIANGLES, indices_.size(), GL_UNSIGNED_INT, 0);
  	glBindVertexArray(0);
}

}  // namespace wvu
