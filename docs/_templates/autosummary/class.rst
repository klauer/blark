{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% set grammar = get_grammar_for_class(objname) %}
   {% if grammar %}
   {% block grammar %}
   .. rubric:: Lark grammar

   This class is used by the following grammar rules:

   {% for name, source in grammar.items() %}
   ``{{ name }}``

   .. code::

     {{ source | indent("     ") }}

   {%- endfor %}
   {% endblock %}
   {% endif %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::

   {% for item in methods %}
   {%- if item not in inherited_members %}
       ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::

   {% for item in attributes %}
       ~{{ name }}.{{ item }}
   {%- endfor %}

   {%- endif %}
   {% endblock %}
