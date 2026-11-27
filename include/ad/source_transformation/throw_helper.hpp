#pragma once

#include "../meta.h"

namespace ad {
consteval void throwKind(std::meta::info i) {
  using namespace ad::meta;
  //if(is_abstract_type(i)) { throw "abstract_type"; }
  /*else*/ if(is_access_specified(i)) { throw "access_specified"; }
  //  else if(is_aggregate_type(i)) { throw "aggregate_type"; }
  else if(is_alias_template(i)) { throw "alias_template"; }
  //  else if(is_arithmetic_type(i)) { throw "arithmetic_type"; }
  //  else if(is_array_type(i)) { throw "array_type"; }
  else if(is_assignment(i)) { throw "assignment"; }
  else if(is_base(i)) { throw "base"; }
  else if(is_operator_expression(i)) { throw "operator_expression"; }
  else if(is_bit_field(i)) { throw "bit_field"; }
  //  else if(is_bounded_array_type(i)) { throw "bounded_array_type"; }
  else if(is_conversion(i)) { throw "conversion"; }
  else if(is_class_member(i)) { throw "class_member"; }
  else if(is_class_template(i)) { throw "class_template"; }
  //  else if(is_class_type(i)) { throw "class_type"; }
  else if(is_complete_type(i)) { throw "complete_type"; }
  else if(is_compound_statement(i)) { throw "compound_statement"; }
  //  else if(is_compound_type(i)) { throw "compound_type"; }
  else if(is_concept(i)) { throw "concept"; }
  else if(is_conditional_operator(i)) { throw "conditional_operator"; }
  else if(is_const(i)) { throw "const"; }
  //  else if(is_const_type(i)) { throw "const_type"; }
  //  else if(is_consteval_only_type(i)) { throw "consteval_only_type"; }
  else if(is_construct(i)) { throw "construct"; }
  else if(is_constructor(i)) { throw "constructor"; }
  else if(is_constructor_template(i)) { throw "constructor_template"; }
  else if(is_conversion_function(i)) { throw "conversion_function"; }
  else if(is_conversion_function_template(i)) { throw "conversion_function_template"; }
  //  else if(is_copy_assignable_type(i)) { throw "copy_assignable_type"; }
  else if(is_copy_assignment(i)) { throw "copy_assignment"; }
  //  else if(is_copy_constructible_type(i)) { throw "copy_constructible_type"; }
  else if(is_copy_constructor(i)) { throw "copy_constructor"; }
  else if(is_data_member_spec(i)) { throw "data_member_spec"; }
  else if(is_declaration_statement(i)) { throw "declaration_statement"; }
  //  else if(is_default_constructible_type(i)) { throw "default_constructible_type"; }
  else if(is_default_constructor(i)) { throw "default_constructor"; }
  else if(is_defaulted(i)) { throw "defaulted"; }
  else if(is_deleted(i)) { throw "deleted"; }
  //  else if(is_destructible_type(i)) { throw "destructible_type"; }
  else if(is_destructor(i)) { throw "destructor"; }
  //  else if(is_empty_type(i)) { throw "empty_type"; }
  //  else if(is_enum_type(i)) { throw "enum_type"; }
  //  else if(is_enumerable_type(i)) { throw "enumerable_type"; }
  else if(is_enumerator(i)) { throw "enumerator"; }
  else if(is_enumerator_spec(i)) { throw "enumerator_spec"; }
  else if(is_explicit(i)) { throw "explicit"; }
  else if(is_explicit_object_parameter(i)) { throw "explicit_object_parameter"; }
  else if(is_expression_statement(i)) { throw "expression_statement"; }
  else if(is_final(i)) { throw "final"; }
  //  else if(is_final_type(i)) { throw "final_type"; }
  //  else if(is_floating_point_type(i)) { throw "floating_point_type"; }
  else if(is_function(i)) { throw "function"; }
  else if(is_function_call(i)) { throw "function_call"; }
  else if(is_function_parameter(i)) { throw "function_parameter"; }
  else if(is_function_template(i)) { throw "function_template"; }
  //  else if(is_function_type(i)) { throw "function_type"; }
  //  else if(is_fundamental_type(i)) { throw "fundamental_type"; }
  //  else if(is_integral_type(i)) { throw "integral_type"; }
  else if(is_literal(i)) { throw "literal"; }
  else if(is_literal_operator(i)) { throw "literal_operator"; }
  else if(is_literal_operator_template(i)) { throw "literal_operator_template"; }
  else if(is_lvalue_reference_qualified(i)) { throw "lvalue_reference_qualified"; }
  //  else if(is_lvalue_reference_type(i)) { throw "lvalue_reference_type"; }
  //  else if(is_member_function_pointer_type(i)) { throw "member_function_pointer_type"; }
  //  else if(is_member_object_pointer_type(i)) { throw "member_object_pointer_type"; }
  //  else if(is_member_pointer_type(i)) { throw "member_pointer_type"; }
  //  else if(is_move_assignable_type(i)) { throw "move_assignable_type"; }
  else if(is_move_assignment(i)) { throw "move_assignment"; }
  //  else if(is_move_constructible_type(i)) { throw "move_constructible_type"; }
  else if(is_move_constructor(i)) { throw "move_constructor"; }
  else if(is_mutable_member(i)) { throw "mutable_member"; }
  else if(is_namespace(i)) { throw "namespace"; }
  else if(is_namespace_alias(i)) { throw "namespace_alias"; }
  else if(is_namespace_member(i)) { throw "namespace_member"; }
  else if(is_noexcept(i)) { throw "noexcept"; }
  else if(is_nonstatic_data_member(i)) { throw "nonstatic_data_member"; }
  //  else if(is_nothrow_copy_assignable_type(i)) { throw "nothrow_copy_assignable_type"; }
  //  else if(is_nothrow_copy_constructible_type(i)) { throw "nothrow_copy_constructible_type"; }
  //  else if(is_nothrow_default_constructible_type(i)) { throw "nothrow_default_constructible_type"; }
  //  else if(is_nothrow_destructible_type(i)) { throw "nothrow_destructible_type"; }
  //  else if(is_nothrow_move_assignable_type(i)) { throw "nothrow_move_assignable_type"; }
  //  else if(is_nothrow_move_constructible_type(i)) { throw "nothrow_move_constructible_type"; }
  //  else if(is_nothrow_swappable_type(i)) { throw "nothrow_swappable_type"; }
  //  else if(is_null_pointer_type(i)) { throw "null_pointer_type"; }
  else if(is_object(i)) { throw "object"; }
  //  else if(is_object_type(i)) { throw "object_type"; }
  else if(is_operator_function(i)) { throw "operator_function"; }
  else if(is_operator_function_template(i)) { throw "operator_function_template"; }
  else if(is_override(i)) { throw "override"; }
  //  else if(is_pointer_type(i)) { throw "pointer_type"; }
  //  else if(is_polymorphic_type(i)) { throw "polymorphic_type"; }
  else if(is_private(i)) { throw "private"; }
  else if(is_protected(i)) { throw "protected"; }
  else if(is_public(i)) { throw "public"; }
  else if(is_pure_virtual(i)) { throw "pure_virtual"; }
  //  else if(is_reference_type(i)) { throw "reference_type"; }
  //  else if(is_reflection_type(i)) { throw "reflection_type"; }
  else if(is_return_statement(i)) { throw "return_statement"; }
  else if(is_rvalue_reference_qualified(i)) { throw "rvalue_reference_qualified"; }
  //  else if(is_rvalue_reference_type(i)) { throw "rvalue_reference_type"; }
  //  else if(is_scalar_type(i)) { throw "scalar_type"; }
  //  else if(is_scoped_enum_type(i)) { throw "scoped_enum_type"; }
  //  else if(is_signed_type(i)) { throw "signed_type"; }
  else if(is_special_member_function(i)) { throw "special_member_function"; }
  //  else if(is_standard_layout_type(i)) { throw "standard_layout_type"; }
  else if(is_statement(i)) { throw "statement"; }
  else if(is_static_member(i)) { throw "static_member"; }
  //  else if(is_structural_type(i)) { throw "structural_type"; }
  else if(is_structured_binding(i)) { throw "structured_binding"; }
  //  else if(is_swappable_type(i)) { throw "swappable_type"; }
  else if(is_template(i)) { throw "template"; }
  //  else if(is_trivial_type(i)) { throw "trivial_type"; }
  //  else if(is_trivially_copy_assignable_type(i)) { throw "trivially_copy_assignable_type"; }
  //  else if(is_trivially_copy_constructible_type(i)) { throw "trivially_copy_constructible_type"; }
  //  else if(is_trivially_copyable_type(i)) { throw "trivially_copyable_type"; }
  //  else if(is_trivially_default_constructible_type(i)) { throw "trivially_default_constructible_type"; }
  //  else if(is_trivially_destructible_type(i)) { throw "trivially_destructible_type"; }
  //  else if(is_trivially_move_assignable_type(i)) { throw "trivially_move_assignable_type"; }
  //  else if(is_trivially_move_constructible_type(i)) { throw "trivially_move_constructible_type"; }
  //  else if(is_type(i)) { throw "type"; }
  else if(is_type_alias(i)) { throw "type_alias"; }
  //  else if(is_unbounded_array_type(i)) { throw "unbounded_array_type"; }
  //  else if(is_union_type(i)) { throw "union_type"; }
  //  else if(is_unsigned_type(i)) { throw "unsigned_type"; }
  else if(is_user_declared(i)) { throw "user_declared"; }
  else if(is_user_provided(i)) { throw "user_provided"; }
  else if(is_value(i)) { throw "value"; }
  else if(is_variable(i)) { throw "variable"; }
  else if(is_id_expression(i)) { throw "id_expression"; }
  else if(is_variable_template(i)) { throw "variable_template"; }
  else if(is_virtual(i)) { throw "virtual"; }
  //  else if(is_void_type(i)) { throw "void_type"; }
  else if(is_volatile(i)) { throw "volatile"; }
  //  else if(is_volatile_type(i)) { throw "volatile_type"; }

  // Generalized ones at the end
  else if(is_expression(i)) { throw "expression"; }

  else { throw "Unkown"; }
}
}