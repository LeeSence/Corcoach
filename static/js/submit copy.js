$(document).ready(function () {
  $('input[type=radio][name="SwitchCheck"]').change(function () {
    console.log($(this).val());
    $("#my_btn").trigger("click");
  });
});
