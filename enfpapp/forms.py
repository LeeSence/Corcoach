from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth import get_user_model

class User_update_form(UserChangeForm):
    class Meta:
        model = get_user_model()
        fields = ('email', 'last_name')